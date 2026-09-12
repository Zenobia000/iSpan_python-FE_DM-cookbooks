from pathlib import Path

import pandas as pd

# 以這支檔案的位置定錨，呼叫端的工作目錄不影響結果。
# utils/ 的上一層就是 data_mining_course/，資料集一律落在其下的 datasets/
# （見 data_setup/data_download.py 的 COURSE_DIR）。
COURSE_DIR = Path(__file__).resolve().parent.parent
DATASETS_DIR = COURSE_DIR / "datasets"

# 各資料集在 datasets/raw/ 底下的實際檔名
_RAW_FILES = {
    'titanic': 'titanic/Titanic-Dataset.csv',
    'house_prices': 'house_prices/train.csv',
    'insurance': 'insurance/insurance.csv',
    'telco_churn': 'telco_churn/WA_Fn-UseC_-Telco-Customer-Churn.csv',
    'mall_customers': 'mall_customers/Mall_Customers.csv',
    'breast_cancer': 'breast_cancer/data.csv',
}


def load_dataset(dataset_name, processed=False):
    """載入指定的資料集"""
    if dataset_name not in _RAW_FILES:
        raise ValueError(f'未知的資料集: {dataset_name}')

    base_path = DATASETS_DIR / ('processed' if processed else 'raw')
    path = base_path / _RAW_FILES[dataset_name]
    if not path.exists():
        raise FileNotFoundError(
            f'找不到 {path}。請先執行 data_setup/data_download.py 下載資料集。'
        )
    return pd.read_csv(path)
