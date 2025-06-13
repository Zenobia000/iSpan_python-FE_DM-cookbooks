import pandas as pd

def load_dataset(dataset_name, processed=False):
    """載入指定的資料集"""
    base_path = '../datasets/'
    if processed:
        base_path += 'processed/'
    else:
        base_path += 'raw/'
    
    # 根據資料集名稱返回對應的資料
    if dataset_name == 'titanic':
        return pd.read_csv(f'{base_path}titanic/train.csv')
    elif dataset_name == 'house_prices':
        return pd.read_csv(f'{base_path}house_prices/train.csv')
    # 其他資料集...
    else:
        raise ValueError(f'未知的資料集: {dataset_name}')
