#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
資料集下載腳本
用於下載資料探勘課程所需的所有 Kaggle 資料集
"""

import os
import subprocess
import time
import shutil
import sys
import zipfile

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
            sys.exit(1)
    
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
            sys.exit(1)

# 檢查並安裝必要的套件
def check_and_install_packages():
    """檢查並安裝必要的套件"""
    required_packages = ['kagglehub', 'requests', 'tqdm']
    
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
                sys.exit(1)

# 定義資料集資訊
def get_datasets_info():
    """獲取所有需要下載的資料集資訊"""
    datasets = [
        {
            "module": "模組三",
            "topic": "缺失值與異常值處理",
            "name": "House Prices",
            "type": "competition",
            "method": "kaggle_cli",
            "competition_id": "house-prices-advanced-regression-techniques",
            "folder": "house_prices"
        },
        {
            "module": "模組四",
            "topic": "類別變數編碼",
            "name": "Titanic",
            "type": "competition",
            "method": "kaggle_cli",
            "competition_id": "titanic",
            "folder": "titanic"
        },
        {
            "module": "模組五",
            "topic": "特徵縮放與變數轉換",
            "name": "Medical Cost Personal Dataset",
            "type": "dataset",
            "method": "kaggle_cli",
            "dataset_id": "mirichoi0218/insurance",
            "folder": "insurance"
        },
        {
            "module": "模組六",
            "topic": "特徵創造",
            "name": "NYC Yellow Taxi Trip Data",
            "type": "dataset",
            "method": "kaggle_cli",
            "dataset_id": "elemento/nyc-yellow-taxi-trip-data",
            "folder": "nyc_taxi",
            "description": "NYC Yellow Taxi Trip Data - Kaggle 資料集"
        },
        {
            "module": "模組七",
            "topic": "特徵選擇與降維",
            "name": "Breast Cancer Wisconsin",
            "type": "dataset",
            "method": "kaggle_cli",
            "dataset_id": "uciml/breast-cancer-wisconsin-data",
            "folder": "breast_cancer"
        },
        {
            "module": "模組八",
            "topic": "時間序列特徵工程",
            "name": "Electric Power Consumption",
            "type": "dataset",
            "method": "kaggle_cli",
            "dataset_id": "uciml/electric-power-consumption-data-set",
            "folder": "power_consumption"
        },
        {
            "module": "模組九",
            "topic": "多模態特徵工程",
            "name": "IMDB 50K Movie Reviews",
            "type": "dataset",
            "method": "kaggle_cli",
            "dataset_id": "lakshmi25npathi/imdb-dataset-of-50k-movie-reviews",
            "folder": "imdb_reviews"
        },
        {
            "module": "模組九",
            "topic": "多模態特徵工程",
            "name": "Dogs vs Cats",
            "type": "competition",
            "method": "kagglehub_only",
            "competition_id": "dogs-vs-cats",
            "folder": "dogs_vs_cats"
        },
        {
            "module": "模組九",
            "topic": "多模態特徵工程",
            "name": "UrbanSound8K",
            "type": "dataset",
            "method": "kaggle_cli",
            "dataset_id": "rupakroy/urban-sound-8k",
            "folder": "urban_sound"
        },
        {
            "module": "模組十",
            "topic": "資料探勘應用",
            "name": "Instacart Market Basket Analysis",
            "type": "dataset",
            "method": "kaggle_cli",
            "dataset_id": "psparks/instacart-market-basket-analysis",
            "folder": "instacart"
        },
        {
            "module": "模組十",
            "topic": "資料探勘應用",
            "name": "Mall Customers",
            "type": "dataset",
            "method": "kaggle_cli",
            "dataset_id": "vjchoudhary7/customer-segmentation-tutorial-in-python",
            "folder": "mall_customers"
        },
        {
            "module": "模組十",
            "topic": "資料探勘應用",
            "name": "Telco Customer Churn",
            "type": "dataset",
            "method": "kaggle_cli",
            "dataset_id": "blastchar/telco-customer-churn",
            "folder": "telco_churn"
        }
    ]
    return datasets

# 原生 Kaggle CLI 下載方法
def download_with_kaggle_cli(dataset, target_folder):
    """使用原生 Kaggle CLI 下載資料集，並自動解壓縮"""
    try:
        is_competition = dataset.get("type") == "competition"
        
        if is_competition:
            # 競賽下載
            entity_id = dataset["competition_id"]
            cmd = ['kaggle', 'competitions', 'download', '-c', entity_id, '--path', target_folder]
        else:
            # 資料集下載
            entity_id = dataset["dataset_id"]
            cmd = ['kaggle', 'datasets', 'download', entity_id, '--path', target_folder, '--unzip']
        
        print(f"   📋 執行命令: {' '.join(cmd)}")
        result = subprocess.run(cmd, capture_output=True)
        
        # Manually decode stdout and stderr with error handling
        stdout_decoded = result.stdout.decode('utf-8', errors='replace')
        stderr_decoded = result.stderr.decode('utf-8', errors='replace')

        if result.returncode == 0:
            print(f"✅ 原生 CLI 下載成功！")

            # 如果是競賽，則手動解壓縮
            if is_competition:
                # 尋找下載的 zip 檔案
                zip_filename = f"{entity_id}.zip"
                zip_filepath = os.path.join(target_folder, zip_filename)

                if os.path.exists(zip_filepath):
                    print(f"   🔄 正在解壓縮檔案: {zip_filename}...")
                    with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                        zip_ref.extractall(target_folder)
                    print(f"   ✅ 解壓縮完成。")
                    os.remove(zip_filepath) # 刪除 zip 檔案
                    print(f"   🗑️  已刪除原始 Zip 檔案。")
                else:
                    # 有些競賽可能不會是標準的 zip 名稱, 比如 titanic
                    for file in os.listdir(target_folder):
                        if file.endswith('.zip'):
                            zip_filepath = os.path.join(target_folder, file)
                            print(f"   🔄 找到並解壓縮檔案: {file}...")
                            with zipfile.ZipFile(zip_filepath, 'r') as zip_ref:
                                zip_ref.extractall(target_folder)
                            os.remove(zip_filepath)
                            print(f"   ✅ 解壓縮完成並已刪除 Zip 檔案。")
                            break

            files = [f for f in os.listdir(target_folder) if not f.endswith('.zip')]
            print(f"📁 目錄中的檔案: {files}")
            return True
        else:
            # 檢查 stderr 和 stdout 中的錯誤信息
            error_msg = stderr_decoded.strip() if stderr_decoded else ""
            output_msg = stdout_decoded.strip() if stdout_decoded else ""
            combined_msg = f"{error_msg} {output_msg}".strip()
            
            if not combined_msg:
                combined_msg = "未知錯誤"
            
            if "401" in combined_msg or "Unauthorized" in combined_msg:
                print(f"❌ CLI 下載失敗 (401 Unauthorized): 請先到 Kaggle 網站接受該競賽/資料集的使用條款。")
                if dataset.get('type') == "competition":
                    print(f"🔗 競賽條款連結: https://www.kaggle.com/c/{dataset.get('competition_id', 'unknown')}")
                else:
                    print(f"🔗 資料集連結: https://www.kaggle.com/datasets/{dataset.get('dataset_id', 'unknown')}")
            elif "403" in combined_msg or "Forbidden" in combined_msg:
                print(f"❌ CLI 下載失敗 (403 Forbidden): 請先到 Kaggle 網站接受該競賽/資料集的使用條款。")
            elif "404" in combined_msg or "Not Found" in combined_msg:
                print(f"❌ CLI 下載失敗 (404 Not Found): 找不到該資料集，請檢查 ID 是否正確。")
            else:
                print(f"❌ 原生 CLI 下載失敗: {combined_msg}")
            return False
            
    except Exception as e:
        print(f"❌ 原生 CLI 下載發生未知錯誤: {str(e)}")
        return False

# KaggleHub 下載方法（改進版 - 基於 simple_download.py 的驗證方法）
def download_with_kagglehub(dataset, target_folder):
    """使用 KaggleHub 下載資料集到指定資料夾"""
    if not KAGGLEHUB_AVAILABLE:
        print("❌ KaggleHub 未安裝")
        return False
    
    try:
        from pathlib import Path
        
        is_competition = dataset.get("type") == "competition"
        
        if is_competition:
            # 競賽下載
            entity_id = dataset["competition_id"]
            print(f"   📋 使用 KaggleHub 下載競賽: {entity_id}")
            cache_path = kagglehub.competition_download(entity_id)
        else:
            # 資料集下載
            entity_id = dataset["dataset_id"]
            print(f"   📋 使用 KaggleHub 下載資料集: {entity_id}")
            cache_path = kagglehub.dataset_download(entity_id)
        
        print(f"✅ KaggleHub 下載成功，緩存路徑: {cache_path}")
        
        # 創建目標資料夾
        target_path = Path(target_folder)
        target_path.mkdir(parents=True, exist_ok=True)
        
        # 複製檔案到目標資料夾
        print(f"   📁 複製檔案到目標目錄: {target_folder}")
        
        # 如果目標資料夾已存在且不為空，先清空
        if target_path.exists() and any(target_path.iterdir()):
            import shutil
            shutil.rmtree(target_path)
            target_path.mkdir(parents=True, exist_ok=True)
        
        # 複製所有檔案
        cache_path_obj = Path(cache_path)
        if cache_path_obj.is_file():
            # 如果是單個檔案
            import shutil
            shutil.copy2(cache_path, target_path / cache_path_obj.name)
        else:
            # 如果是資料夾，複製所有內容
            import shutil
            for item in cache_path_obj.iterdir():
                if item.is_file():
                    shutil.copy2(item, target_path / item.name)
                elif item.is_dir():
                    shutil.copytree(item, target_path / item.name)
        
        print(f"   ✅ 複製完成")
        
        # 顯示下載的檔案
        files = list(target_path.glob('*'))
        print(f"   📄 資料夾中的檔案: {[f.name for f in files]}")
        
        return True
        
    except Exception as e:
        error_msg = str(e)
        if "401" in error_msg or "permission" in error_msg.lower():
            print(f"❌ KaggleHub 下載失敗 (401 Unauthorized): 請先到 Kaggle 網站接受該競賽/資料集的使用條款。")
            if is_competition:
                print(f"🔗 競賽條款連結: https://www.kaggle.com/c/{entity_id}")
            else:
                print(f"🔗 資料集連結: https://www.kaggle.com/datasets/{entity_id}")
        else:
            print(f"❌ KaggleHub 下載失敗: {error_msg}")
        return False

# 全域導入 kagglehub，避免重複導入
try:
    import kagglehub
    KAGGLEHUB_AVAILABLE = True
except ImportError:
    kagglehub = None
    KAGGLEHUB_AVAILABLE = False

# 下載資料集
def download_dataset(dataset, base_dir):
    """下載單個資料集到指定目錄，根據預設方法"""
    from tqdm import tqdm
    
    target_folder = os.path.join(base_dir, "raw", dataset["folder"])
    os.makedirs(target_folder, exist_ok=True)
    
    method = dataset.get("method", "kaggle_cli")
    
    # --- Display Info ---
    print(f"\n{'='*60}")
    print(f"📦 {dataset['name']}")
    print(f"📂 模組: {dataset.get('module', 'N/A')} - {dataset.get('topic', 'N/A')}")
    
    method_map = {
        'kaggle_cli': '💻 Kaggle CLI', 
        'direct': '🌐 直接下載',
        'kagglehub_only': '🤗 KaggleHub'
    }
    print(f"🔧 使用方法: {method_map.get(method, '未知')}")
    print(f"📁 目標資料夾: {target_folder}")
    
    success = False

    # --- Direct Download Method ---
    if method == 'direct':
        with tqdm(total=100, desc="🌐 直接下載", bar_format='{l_bar}{bar}| {percentage:3.0f}%') as pbar:
            try:
                import requests
                url = dataset['direct_url']
                pbar.set_description(f"⬇️ 下載 {os.path.basename(url)}...")
                
                response = requests.get(url, stream=True)
                response.raise_for_status()
                
                total_size = int(response.headers.get('content-length', 0))
                downloaded_size = 0
                
                local_file = os.path.join(target_folder, os.path.basename(url))
                with open(local_file, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            pbar.n = int(progress)
                            pbar.refresh()
                
                pbar.n = 100
                pbar.set_description("✅ 下載完成")
                print(f"\n✅ 成功下載 {dataset['name']} 資料集！")
                print(f"📁 資料集路徑: {local_file}")
                success = True
            except Exception as e:
                pbar.n = 100
                pbar.set_description("❌ 直接下載失敗")
                print(f"\n❌ 直接下載失敗: {str(e)}")
                success = False

        # Try backup URLs if primary failed
        if not success and "backup_urls" in dataset:
            print("\n🔄 主要下載失敗，嘗試備用 URL...")
            for backup_url in dataset["backup_urls"]:
                # 建立一個新的 dataset dict 來遞歸調用
                backup_dataset_info = {
                    "name": f"{dataset['name']} (備用)",
                    "method": "direct",
                    "direct_url": backup_url,
                    "folder": dataset['folder'],
                    "module": dataset.get('module'),
                    "topic": dataset.get('topic')
                }
                if download_dataset(backup_dataset_info, base_dir):
                    success = True
                    break

    # --- Kaggle CLI Method ---
    elif method == 'kaggle_cli':
        with tqdm(total=100, desc="💻 Kaggle CLI", bar_format='{l_bar}{bar}| {percentage:3.0f}%') as pbar:
            pbar.set_description("🔧 準備 CLI 命令..."); pbar.update(20)
            if download_with_kaggle_cli(dataset, target_folder):
                pbar.set_description("✅ CLI 下載完成"); pbar.update(80)
                success = True
            else:
                pbar.set_description("🔄 嘗試 KaggleHub 備用方法"); pbar.update(40)
                if KAGGLEHUB_AVAILABLE and download_with_kagglehub(dataset, target_folder):
                    pbar.set_description("✅ KaggleHub 下載完成"); pbar.update(40)
                    success = True
                else:
                    pbar.set_description("❌ 所有方法都失敗"); pbar.update(20)
                    success = False
    
    # --- KaggleHub Only Method ---
    elif method == 'kagglehub_only':
        with tqdm(total=100, desc="🤗 KaggleHub", bar_format='{l_bar}{bar}| {percentage:3.0f}%') as pbar:
            pbar.set_description("🔧 準備 KaggleHub..."); pbar.update(20)
            if download_with_kagglehub(dataset, target_folder):
                pbar.set_description("✅ KaggleHub 下載完成"); pbar.update(80)
                success = True
            else:
                pbar.set_description("❌ KaggleHub 下載失敗"); pbar.update(80)
                success = False
    
    else:
        print(f"\n❌ 未知的下載方法: {method}")
        success = False

    if not success:
        print(f"\n❌ {dataset['name']} 下載失敗。")
        print("💡 建議檢查:")
        print("   • 網路連線是否正常")
        if method == 'kaggle_cli':
            print("   • Kaggle API 憑證是否正確")
            print("   • 是否已在 Kaggle 網站上手動接受競賽/資料集使用條款")
        if dataset.get('type') == "competition":
             print(f"   • 手動接受條款: https://www.kaggle.com/c/{dataset['competition_id']}")

    return success


def validate_choice(choice, max_value, option_name="選項"):
    """驗證用戶輸入的選擇是否有效"""
    if choice.isdigit() and 1 <= int(choice) <= max_value:
        return True
    print(f"❌ 無效的{option_name}，請輸入 1-{max_value} 之間的數字")
    return False


# 主函數
def main():
    """主函數：檢查環境、下載資料集"""
    print("=" * 60)
    print("資料探勘課程資料集下載工具")
    print("=" * 60)
    
    # 檢查並安裝必要的套件
    check_and_install_packages()
    
    # 檢查 Kaggle API
    check_kaggle_api()
    
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
    print(f"\n📋 可下載的資料集列表 (共 {len(datasets)} 個):")
    print("=" * 80)
    
    for i, dataset in enumerate(datasets, 1):
        method = dataset.get("method", "kaggle_cli")
        method_map = {
            'kaggle_cli': '(CLI+Hub)', 
            'direct': '(Direct)', 
            'kagglehub_only': '(Hub Only)'
        }
        method_tag = method_map.get(method, '')

        if method == "direct":
            dataset_type_icon = "🌐"
            cmd_info = f"直接下載: {dataset['direct_url']}"
        elif method == "kagglehub_only":
            if dataset["type"] == "competition":
                dataset_type_icon = "🤗"
                cmd_info = f"kagglehub.competition_download('{dataset['competition_id']}')"
            else:  # dataset
                dataset_type_icon = "🤗"
                cmd_info = f"kagglehub.dataset_download('{dataset['dataset_id']}')"
        else:  # kaggle_cli
            if dataset["type"] == "competition":
                dataset_type_icon = "🏆"
                cmd_info = f"kaggle competitions download -c {dataset['competition_id']}"
            else:  # dataset
                dataset_type_icon = "📊"
                cmd_info = f"kaggle datasets download {dataset['dataset_id']}"
        
        print(f"{i:2d}. {dataset_type_icon} {method_tag} {dataset['name']}")
        print(f"    📂 模組: {dataset['module']} - {dataset['topic']}")
        print(f"    💻 指令: {cmd_info}")
        print()
    
    print("📌 圖標與標籤說明:")
    print("   🏆 = Kaggle 競賽(CLI), 📊 = Kaggle 資料集(CLI), 🌐 = 直接下載, 🤗 = KaggleHub")
    print("   (CLI+Hub) = Kaggle CLI + KaggleHub 備用, (Direct) = 直接 HTTP 下載, (Hub Only) = 僅 KaggleHub")
    print("=" * 80)
    
    # 提供選項
    print("\n⚙️  請選擇下載選項:")
    print("1. 📦 下載所有資料集（智能下載）")
    print("2. 🎯 下載特定模組的資料集（智能下載）")
    print("3. 🔍 下載單一資料集（智能下載）")
    print("0. ❌ 取消操作")
    print()
    
    choice = input("\n請輸入選項 (0-3): ").strip()
    
    if choice == '1':
        # 下載所有資料集
        from tqdm import tqdm
        
        print(f"\n🚀 開始下載所有 {len(datasets)} 個資料集...")
        success_count = 0
        
        with tqdm(total=len(datasets), desc="📦 總體進度", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} 個資料集') as total_pbar:
            for i, dataset in enumerate(datasets, 1):
                total_pbar.set_description(f"📦 處理 {i}/{len(datasets)}: {dataset['name']}")
                
                if download_dataset(dataset, base_dir):
                    success_count += 1
                    total_pbar.set_postfix({'成功': success_count, '失敗': i - success_count})
                
                total_pbar.update(1)
                time.sleep(1)  # 避免 API 請求過於頻繁
        
        print(f"\n🎉 下載完成！成功下載 {success_count}/{len(datasets)} 個資料集。")
        print(f"📁 資料集已保存在: {raw_dir}")
        
    elif choice == '2':
        # 顯示模組列表
        modules = sorted(list(set([d['module'] for d in datasets])))
        print("\n可選模組:")
        for i, module in enumerate(modules, 1):
            print(f"{i}. {module}")
        
        module_choice = input("\n請選擇模組編號: ").strip()
        if validate_choice(module_choice, len(modules), "模組編號"):
            selected_module = modules[int(module_choice) - 1]
            module_datasets = [d for d in datasets if d['module'] == selected_module]
            
            print(f"\n將下載 {selected_module} 的以下資料集:")
            for i, dataset in enumerate(module_datasets, 1):
                print(f"{i}. {dataset['name']}")
            
            confirm = input("\n確認下載? (y/n): ").strip().lower()
            if confirm == 'y':
                from tqdm import tqdm
                
                print(f"\n🚀 開始下載 {selected_module} 的 {len(module_datasets)} 個資料集...")
                success_count = 0
                
                with tqdm(total=len(module_datasets), desc="📦 模組進度", bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} 個資料集') as module_pbar:
                    for i, dataset in enumerate(module_datasets, 1):
                        module_pbar.set_description(f"📦 處理 {i}/{len(module_datasets)}: {dataset['name']}")
                        
                        if download_dataset(dataset, base_dir):
                            success_count += 1
                            module_pbar.set_postfix({'成功': success_count, '失敗': i - success_count})
                        
                        module_pbar.update(1)
                        time.sleep(1)
                
                print(f"\n🎉 下載完成！成功下載 {success_count}/{len(module_datasets)} 個資料集。")
                print(f"📁 資料集已保存在: {raw_dir}")
            else:
                print("操作已取消。")
    
    elif choice == '3':
        # 下載單一資料集
        dataset_choice = input(f"\n請輸入資料集編號 (1-{len(datasets)}): ").strip()
        if validate_choice(dataset_choice, len(datasets), "資料集編號"):
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
        print("操作已取消。")

if __name__ == "__main__":
    main()