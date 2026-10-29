# 單一機台分析工具範例

這是可供 ds-eda、ds-rules 與 ds-impact 參考的計算範例。完整 harness 的角色、證據與 review 流程見 [docs/harness.md](../../docs/harness.md)。此範例不是任務路由，也不代表所有分析都必須使用相同方法。

在 repository 根目錄使用既有 Python 環境：

```bash
uv sync --locked
uv run --locked python -m examples.factory_analysis demo --out /tmp/factory-demo-001
uv run --locked python -m unittest discover -s examples/factory_analysis/tests -v
```

使用實際資料前複製 `example.config.json` 到本機路徑，設定欄位、事件定義與成本來源，將 `dataset_kind` 改為 `observed`：

```bash
uv run --locked python -m examples.factory_analysis analyze \
  --data /tmp/factory.csv --config /tmp/factory.config.json --out /tmp/factory-run-001
```

輸出目錄必須尚未存在。示範固定亂數種子，資料與成本均為合成或假設。產出包含 proxy、EDA、規則、情境 JSON、圖表與報告。

## 適用限制

資料介面限定單一設備、唯一時間戳記與數值目標。代理特徵限過去至當下的滾動標準差與變化率；滾動視窗按筆數。使用開發集 Spearman 與 BH 校正、深度 2 決策樹及後期資料驗證。缺值不插補，規則分析排除不完整列。

Spearman 與 Wilson 區間以獨立樣本假設近似；時間自相關時需要改用適合的方法。此範例沒有 ANOVA、卡方、群集推論或因果估計。成本情境不證明維護行動會產生該效益。

## 在 harness 中使用

只有資料與方法適用時才執行。技能仍需寫出本次問題對應的分析文件，並用 `record --input` 登錄資料、設定、程式與計算輸出。範例的 `manifest.json` 只記錄範例執行；harness 的 stage、review 與 delivery receipt 才管理這次決策的交接與驗收。
