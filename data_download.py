#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
資料集下載腳本
用於下載資料探勘課程所需的所有 Kaggle 資料集
"""

import os
import subprocess
import pandas as pd
from pathlib import Path
import json
import time
import shutil
import sys

# 確保 Kaggle API 憑證存在
def check_kaggle_api():
    """檢查 Kaggle API 憑證是否存在，若不存在則引導用戶設置"""
    kaggle_dir = os.path.expanduser('~/.kaggle')
    kaggle_api_path = os.path.join(kaggle_dir, 'kaggle.json')
    
    if not os.path.exists(kaggle_api_path):
        print("未找到 Kaggle API 憑證。")
        print("請前往 https://www.kaggle.com/account 獲取 API 憑證")
        print("下載 kaggle.json 檔案後，將其放置於 ~/.kaggle/kaggle.json")
        
        # 嘗試創建目錄
        if not os.path.exists(kaggle_dir):
            os.makedirs(kaggle_dir)
        
        # 檢查當前目錄是否有 kaggle.json
        if os.path.exists('kaggle.json'):
            print("在當前目錄找到 kaggle.json，正在複製到 ~/.kaggle/")
            shutil.copy('kaggle.json', kaggle_api_path)
            os.chmod(kaggle_api_path, 0o600)  # 設置適當的權限
            print(f"已複製 Kaggle API 憑證檔案到: {kaggle_api_path}")
        else:
            print("請將 kaggle.json 放置於當前目錄或 ~/.kaggle/ 目錄")
            exit(1)
    
    # 檢查 kaggle 命令是否可用
    try:
        subprocess.run(['kaggle', '--version'], check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    except (subprocess.SubprocessError, FileNotFoundError):
        print("未安裝 Kaggle CLI。正在嘗試安裝...")
        try:
            subprocess.run(['pip', 'install', 'kaggle'], check=True)
            print("Kaggle CLI 安裝成功！")
        except subprocess.SubprocessError:
            print("安裝 Kaggle CLI 失敗。請手動執行: pip install kaggle")
            exit(1)


# 檢查並安裝必要的套件
def check_and_install_packages():
    """檢查並安裝必要的套件"""
    required_packages = ['kagglehub']
    
    for package in required_packages:
        try:
            __import__(package)
            print(f"✓ {package} 已安裝")
        except ImportError:
            print(f"正在安裝 {package}...")
            try:
                subprocess.run([sys.executable, '-m', 'pip', 'install', package], check=True)
                print(f"✓ {package} 安裝成功")
            except subprocess.SubprocessError:
                print(f"× {package} 安裝失敗，請手動安裝: pip install {package}")
                exit(1)


# 定義資料集資訊
def get_datasets_info():
    """獲取所有需要下載的資料集資訊"""
    datasets = [
        {
            "module": "模組三",
            "topic": "缺失值與異常值處理",
            "name": "House Prices",
            "id": "competitions/house-prices-advanced-regression-techniques",
            "kagglehub_id": "competitions/house-prices-advanced-regression-techniques",
            "folder": "house_prices"
        },
        {
            "module": "模組四",
            "topic": "類別變數編碼",
            "name": "Titanic",
            "id": "competitions/titanic",
            "kagglehub_id": "heptapod/titanic",  # 使用 kagglehub 格式的 ID
            "folder": "titanic"
        },
        {
            "module": "模組五",
            "topic": "特徵縮放與變數轉換",
            "name": "Medical Cost Personal Dataset",
            "id": "datasets/mirichoi0218/insurance",
            "kagglehub_id": "mirichoi0218/insurance",
            "folder": "insurance"
        },
        {
            "module": "模組六",
            "topic": "特徵創造",
            "name": "NYC Taxi Trip Duration",
            "id": "competitions/nyc-taxi-trip-duration",
            "kagglehub_id": "competitions/nyc-taxi-trip-duration",
            "folder": "nyc_taxi"
        },
        {
            "module": "模組七",
            "topic": "特徵選擇與降維",
            "name": "Breast Cancer Wisconsin",
            "id": "datasets/uciml/breast-cancer-wisconsin-data",
            "kagglehub_id": "uciml/breast-cancer-wisconsin-data",
            "folder": "breast_cancer"
        },
        {
            "module": "模組八",
            "topic": "時間序列特徵工程",
            "name": "Electric Power Consumption",
            "id": "datasets/uciml/electric-power-consumption-data-set",
            "kagglehub_id": "uciml/electric-power-consumption-data-set",
            "folder": "power_consumption"
        },
        {
            "module": "模組九",
            "topic": "多模態特徵工程",
            "name": "IMDB 50K Movie Reviews",
            "id": "datasets/lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
            "kagglehub_id": "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
            "folder": "imdb_reviews"
        },
        {
            "module": "模組九",
            "topic": "多模態特徵工程",
            "name": "Dogs vs Cats",
            "id": "competitions/dogs-vs-cats",
            "kagglehub_id": "competitions/dogs-vs-cats",
            "folder": "dogs_vs_cats"
        },
        {
            "module": "模組九",
            "topic": "多模態特徵工程",
            "name": "UrbanSound8K",
            "id": "datasets/rupakroy/urban-sound-8k",
            "kagglehub_id": "rupakroy/urban-sound-8k",
            "folder": "urban_sound"
        },
        {
            "module": "模組十",
            "topic": "資料探勘應用",
            "name": "Instacart Market Basket Analysis",
            "id": "competitions/instacart-market-basket-analysis",
            "kagglehub_id": "competitions/instacart-market-basket-analysis",
            "folder": "instacart"
        },
        {
            "module": "模組十",
            "topic": "資料探勘應用",
            "name": "Mall Customers",
            "id": "datasets/vjchoudhary7/customer-segmentation-tutorial-in-python",
            "kagglehub_id": "vjchoudhary7/customer-segmentation-tutorial-in-python",
            "folder": "mall_customers"
        },
        {
            "module": "模組十",
            "topic": "資料探勘應用",
            "name": "Telco Customer Churn",
            "id": "datasets/blastchar/telco-customer-churn",
            "kagglehub_id": "blastchar/telco-customer-churn",
            "folder": "telco_churn"
        }
    ]
    return datasets

# 下載資料集
def download_dataset(dataset, base_dir):
    """使用 kagglehub 下載單個資料集到指定目錄"""
    kagglehub_id = dataset["kagglehub_id"]
    target_folder = os.path.join(base_dir, "raw", dataset["folder"])
    
    # 創建目標資料夾
    os.makedirs(target_folder, exist_ok=True)
    
    print(f"\n正在下載 {dataset['name']} 資料集...")
    
    try:
        # 使用 kagglehub 下載資料集
        import kagglehub
        path = kagglehub.dataset_download(kagglehub_id, target_folder)
        
        print(f"成功下載 {dataset['name']} 資料集！")
        print(f"資料集路徑: {path}")
        return True
        
    except Exception as e:
        print(f"下載 {dataset['name']} 資料集時發生錯誤: {str(e)}")
        
        # 特殊處理 Titanic 資料集
        if dataset["name"] == "Titanic":
            try:
                print("嘗試從備用來源下載 Titanic 資料集...")
                import requests
                
                # 從 GitHub 下載 Titanic 資料集
                titanic_url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv"
                response = requests.get(titanic_url)
                
                if response.status_code == 200:
                    csv_path = os.path.join(target_folder, "titanic.csv")
                    with open(csv_path, 'wb') as f:
                        f.write(response.content)
                    
                    print(f"成功從備用來源下載 Titanic 資料集！")
                    print(f"資料集路徑: {csv_path}")
                    return True
                else:
                    print(f"從備用來源下載失敗，狀態碼: {response.status_code}")
            
            except Exception as e2:
                print(f"從備用來源下載 Titanic 資料集時發生錯誤: {str(e2)}")
        
        print(f"下載 {dataset['name']} 資料集失敗。")
        return False

# 主函數
def main():
    """主函數：檢查環境、下載資料集"""
    print("=" * 60)
    print("資料探勘課程資料集下載工具")
    print("=" * 60)
    
    # 檢查並安裝必要的套件
    check_and_install_packages()
    
    # 設置資料目錄
    base_dir = os.path.join(os.getcwd(), "datasets")
    raw_dir = os.path.join(base_dir, "raw")
    processed_dir = os.path.join(base_dir, "processed")
    
    # 創建必要的目錄
    os.makedirs(raw_dir, exist_ok=True)
    os.makedirs(processed_dir, exist_ok=True)
    
    # 獲取資料集資訊
    datasets = get_datasets_info()
    
    # 顯示將要下載的資料集
    print(f"\n可下載的資料集列表 (共 {len(datasets)} 個):")
    for i, dataset in enumerate(datasets, 1):
        print(f"{i}. {dataset['name']} ({dataset['module']}: {dataset['topic']})")
    
    # 提供選項
    print("\n請選擇下載選項:")
    print("1. 下載所有資料集")
    print("2. 下載特定模組的資料集")
    print("3. 下載單一資料集")
    print("0. 取消操作")
    
    choice = input("\n請輸入選項 (0-3): ").strip()
    
    if choice == '1':
        # 下載所有資料集
        success_count = 0
        for dataset in datasets:
            if download_dataset(dataset, base_dir):
                success_count += 1
            time.sleep(1)  # 避免 API 請求過於頻繁
        
        print(f"\n下載完成！成功下載 {success_count}/{len(datasets)} 個資料集。")
        print(f"資料集已保存在: {raw_dir}")
        
    elif choice == '2':
        # 顯示模組列表
        modules = sorted(list(set([d['module'] for d in datasets])))
        print("\n可選模組:")
        for i, module in enumerate(modules, 1):
            print(f"{i}. {module}")
        
        module_choice = input("\n請選擇模組編號: ").strip()
        if module_choice.isdigit() and 1 <= int(module_choice) <= len(modules):
            selected_module = modules[int(module_choice) - 1]
            module_datasets = [d for d in datasets if d['module'] == selected_module]
            
            print(f"\n將下載 {selected_module} 的以下資料集:")
            for i, dataset in enumerate(module_datasets, 1):
                print(f"{i}. {dataset['name']}")
            
            confirm = input("\n確認下載? (y/n): ").strip().lower()
            if confirm == 'y':
                success_count = 0
                for dataset in module_datasets:
                    if download_dataset(dataset, base_dir):
                        success_count += 1
                    time.sleep(1)
                
                print(f"\n下載完成！成功下載 {success_count}/{len(module_datasets)} 個資料集。")
                print(f"資料集已保存在: {raw_dir}")
            else:
                print("操作已取消。")
        else:
            print("無效的選擇，操作已取消。")
    
    elif choice == '3':
        # 下載單一資料集
        dataset_choice = input("\n請輸入資料集編號 (1-{}): ".format(len(datasets))).strip()
        if dataset_choice.isdigit() and 1 <= int(dataset_choice) <= len(datasets):
            dataset = datasets[int(dataset_choice) - 1]
            print(f"\n將下載: {dataset['name']} ({dataset['module']}: {dataset['topic']})")
            
            confirm = input("確認下載? (y/n): ").strip().lower()
            if confirm == 'y':
                if download_dataset(dataset, base_dir):
                    print(f"\n成功下載 {dataset['name']} 資料集！")
                    print(f"資料集已保存在: {os.path.join(raw_dir, dataset['folder'])}")
                else:
                    print(f"\n下載 {dataset['name']} 資料集失敗。")
            else:
                print("操作已取消。")
        else:
            print("無效的選擇，操作已取消。")
    
    else:
        print("操作已取消。")

if __name__ == "__main__":
    main()