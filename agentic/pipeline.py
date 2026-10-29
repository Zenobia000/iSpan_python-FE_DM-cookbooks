"""Deterministic evidence tools; agents own reasoning and interpretation."""

import hashlib
import json
import platform
from importlib.metadata import version
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.tree import DecisionTreeClassifier
from statsmodels.stats.multitest import multipletests


def write_json(path, value):
    Path(path).write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def finite_number(value, label, minimum=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{label} 必須為數值")
    if not np.isfinite(value) or (minimum is not None and value < minimum):
        raise ValueError(f"{label} 數值超出範圍")
    return value


def add_proxies(frame, specs, target, timestamp):
    result = frame.copy()
    names = set(frame.columns)
    for spec in specs:
        name, source = spec["name"], spec["source"]
        if name in names or source in (target, timestamp) or source not in frame:
            raise ValueError("代理特徵名稱重複、來源不存在或使用目標／時間欄位")
        if not spec.get("rationale") or spec.get("available_at_decision") is not True:
            raise ValueError("代理特徵須說明領域理由，並確認決策當下可取得")
        values = pd.to_numeric(frame[source], errors="raise")
        if spec["operation"] == "rolling_std":
            window = spec["window"]
            if isinstance(window, bool) or not isinstance(window, int) or window < 2:
                raise ValueError("rolling_std window 必須為至少 2 的整數")
            result[name] = values.rolling(window, min_periods=window).std()
        elif spec["operation"] == "pct_change":
            result[name] = values.pct_change(fill_method=None)
        else:
            raise ValueError("僅支援 rolling_std 與 pct_change，不執行任意表達式")
        result[name] = result[name].replace([np.inf, -np.inf], np.nan)
        names.add(name)
    return result


def associations(frame, target, features):
    records = []
    for feature in features:
        pair = frame[[feature, target]].dropna()
        if len(pair) < 10 or pair[feature].nunique() < 2 or pair[target].nunique() < 2:
            continue
        rho, p = spearmanr(pair[feature], pair[target])
        if not np.isfinite(rho) or not np.isfinite(p):
            continue
        records.append({"feature": feature, "n": len(pair), "spearman_rho": float(rho),
                        "p_value": float(p)})
    if records:
        q_values = multipletests([r["p_value"] for r in records], method="fdr_bh")[1]
        for record, q in zip(records, q_values):
            record["q_value_bh"] = float(q)
        records.sort(key=lambda r: (r["q_value_bh"], -abs(r["spearman_rho"])))
    return records


def wilson_interval(positive, n):
    if not n:
        return None
    z = 1.96
    p = positive / n
    denominator = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denominator
    margin = z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denominator
    return [max(0.0, float(center - margin)), min(1.0, float(center + margin))]


def leaf_evidence(leaf_ids, labels, leaf):
    selected = labels[leaf_ids == leaf]
    n, positive = int(len(selected)), int(selected.sum())
    return {"n": n, "coverage": n / len(labels), "events": positive,
            "event_rate": positive / n if n else None,
            "event_rate_wilson_95": wilson_interval(positive, n)}


def extract_rules(train, holdout, target, features, threshold, min_leaf):
    train = train[features + [target]].dropna()
    holdout = holdout[features + [target]].dropna()
    if len(train) < 2 * min_leaf or len(holdout) < 10:
        raise ValueError("完整特徵樣本不足：開發集至少 2 × min_leaf，驗證集至少 10 筆")
    y_train = (train[target] < threshold).astype(int).to_numpy()
    y_holdout = (holdout[target] < threshold).astype(int).to_numpy()
    if len(np.unique(y_train)) < 2:
        raise ValueError("開發集須同時包含正常與異常事件")
    model = DecisionTreeClassifier(max_depth=2, min_samples_leaf=min_leaf, random_state=42)
    model.fit(train[features], y_train)
    train_ids, holdout_ids = model.apply(train[features]), model.apply(holdout[features])
    rules = []

    def walk(node, conditions):
        tree = model.tree_
        if tree.children_left[node] == tree.children_right[node]:
            rules.append({"conditions": conditions, "status": "candidate",
                          "train": leaf_evidence(train_ids, y_train, node),
                          "holdout": leaf_evidence(holdout_ids, y_holdout, node)})
            return
        feature = features[tree.feature[node]]
        cut = float(tree.threshold[node])
        walk(tree.children_left[node], conditions + [{"feature": feature, "op": "<=", "value": cut}])
        walk(tree.children_right[node], conditions + [{"feature": feature, "op": ">", "value": cut}])

    walk(0, [])
    predictions = model.predict(holdout[features])
    tn = int(((predictions == 0) & (y_holdout == 0)).sum())
    fp = int(((predictions == 1) & (y_holdout == 0)).sum())
    fn = int(((predictions == 0) & (y_holdout == 1)).sum())
    tp = int(((predictions == 1) & (y_holdout == 1)).sum())
    return {"event_definition": f"{target} < {threshold}", "max_depth": 2,
            "min_samples_leaf": min_leaf, "train_n": len(train), "holdout_n": len(holdout),
            "holdout_event_rate": float(y_holdout.mean()),
            "holdout_confusion": {"tn": tn, "fp": fp, "fn": fn, "tp": tp}, "rules": rules}


def business_scenarios(costs):
    monthly_loss = finite_number(costs["monthly_baseline_loss"], "monthly_baseline_loss", 0)
    monthly_cost = finite_number(costs["monthly_action_cost"], "monthly_action_cost", 0)
    initial = finite_number(costs["initial_cost"], "initial_cost", 0)
    months = finite_number(costs["months"], "months", 1)
    investment = initial + monthly_cost * months
    scenarios = []
    for fraction in costs["assumed_reductions"]:
        finite_number(fraction, "assumed_reductions", 0)
        if fraction > 1:
            raise ValueError("成本減少比例須在 0 到 1 之間")
        avoided = monthly_loss * months * fraction
        net = avoided - investment
        scenarios.append({"assumed_reduction": fraction, "avoided_loss": avoided,
                          "investment": investment, "net_benefit": net,
                          "roi": net / investment if investment else None})
    return {"status": "scenario_not_measured", "currency": costs["currency"],
            "months": months, "inputs": costs, "scenarios": scenarios}


def run(data_path, config_path, out):
    out = Path(out)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    frame = pd.read_csv(data_path)
    target, timestamp = config["target"], config["timestamp"]
    features = config["features"]
    fraction = finite_number(config["train_fraction"], "train_fraction")
    threshold = finite_number(config["event_below"], "event_below")
    if not 0.5 <= fraction <= 0.9:
        raise ValueError("train_fraction 須介於 0.5 與 0.9")
    min_leaf = config["min_samples_leaf"]
    if isinstance(min_leaf, bool) or not isinstance(min_leaf, int) or min_leaf < 5:
        raise ValueError("min_samples_leaf 須為至少 5 的整數")
    if not isinstance(features, list) or not features or len(set(features)) != len(features):
        raise ValueError("features 須為非空且不重複的欄位清單")
    if target == timestamp or target in features or timestamp in features:
        raise ValueError("目標、時間與分析特徵不可重疊")
    times = pd.to_datetime(frame[timestamp], errors="raise", utc=True)
    if times.isna().any() or times.duplicated().any():
        raise ValueError("目前工具適用單一設備資料，時間不得缺失或重複")
    frame = frame.assign(**{timestamp: times}).sort_values(timestamp).reset_index(drop=True)
    boundary = int(len(frame) * fraction)
    if boundary < 20 or len(frame) - boundary < 10:
        raise ValueError("時間切分後開發集至少須有 20 筆，驗證集至少須有 10 筆")
    # Fix the time boundary before constructing causal, trailing-only features.
    frame = add_proxies(frame, config["proxies"], target, timestamp)
    for column in features + [target]:
        frame[column] = pd.to_numeric(frame[column], errors="raise")
        if np.isinf(frame[column]).any():
            raise ValueError(f"{column} 包含無窮大")
    if frame[target].isna().any():
        raise ValueError("目標值不可缺失")
    train, holdout = frame.iloc[:boundary], frame.iloc[boundary:]
    eda = associations(train, target, features)
    rules = extract_rules(train, holdout, target, features, threshold, min_leaf)
    impact = business_scenarios(config["business"])
    proxies = [{**spec, "status": "hypothesis_not_domain_validated"} for spec in config["proxies"]]
    write_json(out / "proxies.json", proxies)
    write_json(out / "eda.json", eda)
    write_json(out / "rules.json", rules)
    write_json(out / "impact.json", impact)
    for index, record in enumerate(eda[:5], 1):
        feature = record["feature"]
        fig, ax = plt.subplots(figsize=(6, 4))
        ax.scatter(train[feature], train[target], s=10, alpha=0.5)
        ax.set(xlabel=feature, ylabel=target, title="Development data only")
        fig.tight_layout()
        fig.savefig(out / f"eda-{index}.png", dpi=120)
        plt.close(fig)
    manifest = {
        "status": "completed", "dataset_kind": config["dataset_kind"],
        "input_sha256": hashlib.sha256(Path(data_path).read_bytes()).hexdigest(),
        "config_sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest(),
        "config": config, "python": platform.python_version(),
        "versions": {name: version(name) for name in
                     ["numpy", "pandas", "scipy", "scikit-learn", "statsmodels", "matplotlib"]},
        "split": {"train_rows": len(train), "holdout_rows": len(holdout),
                  "train_end": train[timestamp].iloc[-1].isoformat(),
                  "holdout_start": holdout[timestamp].iloc[0].isoformat()},
        "missing_feature_rows": {"train": int(train[features].isna().any(axis=1).sum()),
                                 "holdout": int(holdout[features].isna().any(axis=1).sum())},
        "artifacts": sorted({p.name for p in out.iterdir()} | {"report.md", "handoff.md", "manifest.json"}),
    }
    report = ["# 決策分析證據報告", "", f"資料類型：{config['dataset_kind']}。", "",
              "## 代理特徵", "", "候選特徵與理由見 proxies.json；尚未完成領域驗證。", "",
              "## 開發集探索", "", "Spearman 效果量、樣本數與 BH 校正見 eda.json。",
              "p 與 q 值以獨立樣本假設近似；時間自相關或群集資料須另用適合的推論方法。",
              "圖表依校正結果與相關幅度排序，未設定業務效果門檻，不代表前五項都值得採用。", ""]
    for index, record in enumerate(eda[:5], 1):
        report.extend([f"![開發集關聯圖 {index}](eda-{index}.png)", ""])
    report.extend(["## 候選規則", "", f"異常定義：{rules['event_definition']}。",
                   f"驗證集完整樣本 {rules['holdout_n']} 筆；異常比例 {rules['holdout_event_rate']:.3f}。",
                   "各葉節點的條件、涵蓋率、事件比例與 Wilson 區間見 rules.json。",
                   "區間以二項獨立事件假設計算；樣本相依時可能低估不確定性。",
                   "驗證結果供評估候選規則，現場 SOP 與警報動作仍待領域確認。", "",
                   "| 候選條件 | 開發樣本 | 驗證樣本 | 驗證異常比例 |",
                   "| --- | --- | --- | --- |"])
    for rule in rules["rules"]:
        condition = " 且 ".join(
            f"{item['feature']} {item['op']} {item['value']:.17g}"
            for item in rule["conditions"]
        ) or "全部樣本"
        rate = rule["holdout"]["event_rate"]
        displayed_rate = "無樣本" if rate is None else f"{rate:.1%}"
        report.append(f"| {condition} | {rule['train']['n']} | {rule['holdout']['n']} | {displayed_rate} |")
    report.extend(["", "## 商業情境", "", "以下均為假設情境，維護提前時間與改善幅度之因果效果尚未驗證。", "",
                   "| 假設成本減少 | 可避免損失 | 總投入 | 淨效益 | ROI |",
                   "| --- | --- | --- | --- | --- |"])
    for scenario in impact["scenarios"]:
        roi = "不適用" if scenario["roi"] is None else f"{scenario['roi']:.1%}"
        report.append(f"| {scenario['assumed_reduction']:.0%} | {scenario['avoided_loss']:.2f} | "
                      f"{scenario['investment']:.2f} | {scenario['net_benefit']:.2f} | {roi} |")
    report.extend(["", f"幣別：{impact['currency']}；評估期間：{impact['months']} 個月。", "",
                   "## 下一步", "", "由領域專家核對代理特徵、候選規則與成本參數，選定試行範圍。",
                   "本工具不會依驗證集分數自動調整規則或自動部署。", ""])
    (out / "report.md").write_text("\n".join(report), encoding="utf-8")
    (out / "handoff.md").write_text(
        "# 分析交接\n\n已完成四環節的可重現計算，證據與設定見 manifest.json。\n\n"
        "待完成：領域假設確認、序列相依性檢查、業務效果門檻、規則處置設計與現場試行。\n\n"
        "下一位 agent：先讀 report.md 與 manifest.json，記錄你的解讀與使用者回覆。"
        "若修改假設或參數，使用新的產出目錄；不得反覆利用同一驗證集調參後宣稱獨立驗證。\n",
        encoding="utf-8",
    )
    write_json(out / "manifest.json", manifest)
    return out


def make_demo(out):
    rng = np.random.default_rng(42)
    n = 400
    temperature = rng.normal(80, 7, n)
    pressure = rng.normal(1.4, 0.3, n)
    stress = (temperature > 85) & (pressure < 1.2)
    frame = pd.DataFrame({
        "timestamp": pd.date_range("2026-01-01", periods=n, freq="h", tz="UTC"),
        "temperature": temperature, "pressure": pressure,
        "motor_current": rng.normal(10, 1, n) + stress * rng.normal(3, 1, n),
        "power_kwh": rng.uniform(20, 30, n) + stress * 8,
        "yield_rate": np.clip(0.97 - stress * 0.15 + rng.normal(0, 0.012, n), 0, 1),
    })
    config = json.loads((Path(__file__).parent / "example.config.json").read_text(encoding="utf-8"))
    data_path, config_path = Path(out) / "synthetic.csv", Path(out) / "config.json"
    frame.to_csv(data_path, index=False)
    write_json(config_path, config)
    return data_path, config_path
