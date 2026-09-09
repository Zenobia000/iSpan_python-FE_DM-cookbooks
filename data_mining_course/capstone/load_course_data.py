"""總整（英國二手車）資料載入。只給 capstone 用，不是全課資料入口。"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

def setup_plotting() -> None:
    import matplotlib.pyplot as plt

    plt.style.use("seaborn-v0_8")
    plt.rcParams["font.sans-serif"] = [
        "Microsoft JhengHei",
        "Microsoft YaHei",
        "Noto Sans CJK TC",
        "SimHei",
        "DejaVu Sans",
    ]
    plt.rcParams["axes.unicode_minus"] = False


def load_unclean_preview(n: int = 8) -> tuple[pd.DataFrame, str]:
    path = CAR_DIR / "unclean cclass.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path).head(n), f"未清理檔（品質對照）{path}"


TRACK_DIR = Path(__file__).resolve().parent
COURSE_DIR = TRACK_DIR.parent
REPO_DIR = COURSE_DIR.parent

CAR_DIR = COURSE_DIR / "projects" / "project" / "car_data"
# data_setup/data_download.py 現在以檔案位置定錨，資料一律落在 data_mining_course/datasets/，
# 但舊版是以「執行時的工作目錄」為根，既有的下載可能還留在 repo 根目錄。兩處都列入候選，
# 避免明明下載過卻走進合成資料的後備路徑。
_DATASET_ROOTS = [COURSE_DIR / "datasets" / "raw", REPO_DIR / "datasets" / "raw"]


def _dataset_candidates(*parts: str) -> list[Path]:
    return [root.joinpath(*parts) for root in _DATASET_ROOTS]


TELCO_CANDIDATES = [
    COURSE_DIR / "modules" / "module_10_data_mining_applications" / "datasets" / "raw" / "telco_churn" / "WA_Fn-UseC_-Telco-Customer-Churn.csv",
    *_dataset_candidates("telco_churn", "WA_Fn-UseC_-Telco-Customer-Churn.csv"),
]
HOUSE_CANDIDATES = _dataset_candidates("house_prices", "train.csv")
INSURANCE_CANDIDATES = _dataset_candidates("insurance", "insurance.csv")
TAXI_CANDIDATES = _dataset_candidates("nyc_taxi", "train.csv")
POWER_CANDIDATES = _dataset_candidates("power_consumption", "power_consumption.csv")

BRAND_FILES = {
    "audi": "Audi",
    "bmw": "BMW",
    "ford": "Ford",
    "hyundi": "Hyundai",
    "merc": "Mercedes",
    "skoda": "Skoda",
    "toyota": "Toyota",
    "vauxhall": "Vauxhall",
    "vw": "VW",
}
SNAPSHOT_YEAR = 2020


def _first_existing(paths: list[Path]) -> Path | None:
    for path in paths:
        if path.exists():
            return path
    return None


def load_used_cars(sample_n: int = 12000, random_state: int = 42) -> tuple[pd.DataFrame, str]:
    if not CAR_DIR.exists():
        raise FileNotFoundError(f"找不到二手車資料：{CAR_DIR}")
    parts = []
    for fname, brand in BRAND_FILES.items():
        path = CAR_DIR / f"{fname}.csv"
        chunk = pd.read_csv(path)
        chunk = chunk.rename(columns={"tax(£)": "tax"})
        chunk["brand"] = brand
        parts.append(chunk)
    raw = pd.concat(parts, ignore_index=True)
    if sample_n and len(raw) > sample_n:
        raw = raw.sample(n=sample_n, random_state=random_state)
    note = f"英國二手車（你的 Kaggle 預錄）{CAR_DIR}，抽樣 {len(raw):,} 列。"
    return raw.reset_index(drop=True), note


def load_telco() -> tuple[pd.DataFrame, str]:
    path = _first_existing(TELCO_CANDIDATES)
    if path is not None:
        df = pd.read_csv(path)
        df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
        return df, f"Telco Churn（你的 Kaggle 預錄）{path}"

    rng = np.random.default_rng(1)
    n = 4000
    tenure = rng.integers(0, 73, n)
    monthly = rng.normal(65, 20, n).clip(18, 120)
    total = np.where(tenure == 0, np.nan, monthly * tenure + rng.normal(0, 40, n))
    contract = rng.choice(["Month-to-month", "One year", "Two year"], n, p=[0.55, 0.24, 0.21])
    internet = rng.choice(["DSL", "Fiber optic", "No"], n, p=[0.34, 0.44, 0.22])
    logit = (
        -1.1
        + 1.35 * (contract == "Month-to-month")
        - 0.028 * tenure
        + 0.012 * monthly
        + 0.35 * (internet == "Fiber optic")
    )
    churn = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
    df = pd.DataFrame(
        {
            "customerID": [f"C{i:05d}" for i in range(n)],
            "tenure": tenure,
            "MonthlyCharges": monthly.round(2),
            "TotalCharges": np.round(total, 2),
            "Contract": contract,
            "InternetService": internet,
            "Churn": np.where(churn == 1, "Yes", "No"),
        }
    )
    note = (
        "磁碟沒有 Telco CSV，改用可重現的合成流失表示範樹重要性與缺值。"
        "把 WA_Fn-UseC_-Telco-Customer-Churn.csv 放到 datasets/raw/telco_churn/ 即會改走原檔。"
    )
    return df, note


def load_house_prices() -> tuple[pd.DataFrame, str]:
    path = _first_existing(HOUSE_CANDIDATES)
    if path is not None:
        return pd.read_csv(path), f"House Prices {path}"
    cars, _ = load_used_cars(sample_n=8000)
    demo = cars.copy()
    rng = np.random.default_rng(0)
    mask = rng.random(len(demo)) < 0.12
    demo.loc[mask, "tax"] = np.nan
    note = (
        "磁碟沒有 House Prices CSV，改用二手車模擬「缺值可以是訊號」："
        "隨機挖空 tax。把 train.csv 放到 datasets/raw/house_prices/ 即會改走原檔。"
    )
    return demo, note


def load_insurance_like() -> tuple[pd.DataFrame, str]:
    path = _first_existing(INSURANCE_CANDIDATES)
    if path is not None:
        return pd.read_csv(path), f"Insurance {path}"
    cars, _ = load_used_cars(sample_n=8000)
    demo = cars[["price", "mileage", "mpg", "engineSize", "tax"]].copy()
    note = (
        "磁碟沒有 Insurance CSV，改用二手車數值欄示範縮放。"
        "把 insurance.csv 放到 datasets/raw/insurance/ 即會改走原檔。"
    )
    return demo, note


def load_taxi_or_cars_for_time() -> tuple[pd.DataFrame, str]:
    path = _first_existing(TAXI_CANDIDATES)
    if path is not None:
        df = pd.read_csv(path)
        return df, f"NYC Taxi {path}"
    cars, _ = load_used_cars(sample_n=8000)
    demo = cars.copy()
    demo["list_date"] = pd.to_datetime(
        dict(year=demo["year"].clip(upper=SNAPSHOT_YEAR), month=6, day=15)
    )
    note = (
        "磁碟沒有 NYC Taxi train.csv，改用二手車 year 當時間欄示範日曆特徵與聚合。"
        "把 train.csv 放到 datasets/raw/nyc_taxi/ 即會改走原檔。"
    )
    return demo, note


def make_power_series(periods: int = 24 * 60) -> tuple[pd.DataFrame, str]:
    path = _first_existing(POWER_CANDIDATES)
    if path is not None:
        df = pd.read_csv(path)
        return df, f"電力消耗 {path}"
    rng = np.random.default_rng(7)
    idx = pd.date_range("2024-01-01", periods=periods, freq="h")
    hour = idx.hour
    weekly = idx.dayofweek
    load = (
        800
        + 180 * np.sin(2 * np.pi * hour / 24)
        + 90 * (weekly < 5).astype(float)
        + rng.normal(0, 35, size=periods)
    )
    df = pd.DataFrame({"timestamp": idx, "load_mw": load})
    note = (
        "磁碟沒有電力消耗 CSV，改用可重現的小時負載序列示範 lag/rolling。"
        "把檔案放到 datasets/raw/power_consumption/ 即會改走原檔。"
    )
    return df, note
