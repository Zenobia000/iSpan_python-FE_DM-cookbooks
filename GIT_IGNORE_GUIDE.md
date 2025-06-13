# Git 忽略檔案指南

## 概述

由於專案中已經有 `data_download.py` 腳本可以自動下載所有需要的 Kaggle 資料集，因此大型資料檔案不需要納入版本控制。

## 不需要上傳的檔案類型

### 🔴 資料集檔案 (最重要)
```
datasets/raw/          # 原始資料集
datasets/processed/    # 處理後的資料集
data_mining_course/datasets/raw/
data_mining_course/datasets/processed/
```

**原因:**
- 檔案過大，會拖慢 Git 操作
- 可透過 `data_download.py` 重新下載
- Kaggle 資料集會不斷更新

### 🔴 敏感資訊檔案
```
kaggle.json           # Kaggle API 憑證
.kaggle/              # Kaggle 配置目錄
.env                  # 環境變數
secrets.json          # 機密設定
credentials.json      # 登入憑證
```

**原因:**
- 包含個人 API 金鑰和密碼
- 洩露會造成安全風險

### 🟡 暫存和快取檔案
```
__pycache__/          # Python 快取
.ipynb_checkpoints/   # Jupyter 檢查點
*.pyc                 # Python 編譯檔案
.mypy_cache/          # 型別檢查快取
```

**原因:**
- 系統自動產生
- 在不同環境中會重新建立

### 🟡 虛擬環境
```
venv/                 # Python 虛擬環境
env/
.venv/
conda-env/
```

**原因:**
- 包含大量第三方套件
- 應該由 requirements.txt 管理

### 🟡 模型和實驗檔案
```
*.pth                 # PyTorch 模型
*.h5                  # Keras/HDF5 模型
*.pkl                 # Pickle 檔案
checkpoints/          # 訓練檢查點
runs/                 # TensorBoard 日誌
experiments/          # 實驗結果
```

**原因:**
- 檔案通常很大 (數 GB)
- 可透過訓練腳本重新產生

### 🟡 個人檔案
```
.Clipboard_Record.md  # 個人剪貼簿記錄
*.log                 # 日誌檔案
.DS_Store            # macOS 系統檔案
Thumbs.db            # Windows 縮圖快取
```

## 資料下載流程

由於資料集不上傳到 Git，新使用者需要：

1. **設置 Kaggle API:**
   ```bash
   # 下載 kaggle.json 到 ~/.kaggle/
   mkdir ~/.kaggle
   cp kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json
   ```

2. **執行下載腳本:**
   ```bash
   python data_download.py
   ```

3. **選擇下載選項:**
   - 下載所有資料集
   - 下載特定模組資料集
   - 下載單一資料集

## 專案大小優化

### 💡 優化前 vs 優化後
- **優化前:** 可能 10+ GB (包含所有資料集)
- **優化後:** < 100 MB (僅程式碼和文件)

### 💡 優點
- ✅ 快速 clone 和 push
- ✅ 節省 GitHub 儲存空間
- ✅ 避免版本衝突
- ✅ 保護敏感資訊
- ✅ 便於多人協作

## 例外情況

如果需要包含特定的小型範例檔案，可以在 `.gitignore` 中使用 `!` 排除：

```gitignore
# 忽略所有 CSV 檔案
*.csv

# 但保留特定的範例檔案
!examples/sample_data.csv
!templates/example_output.csv
```

## 注意事項

1. **初次使用者:** 需要自行設置 Kaggle API 憑證
2. **資料更新:** 定期執行 `data_download.py` 獲取最新資料
3. **大型檔案:** 如需版本控制，考慮使用 Git LFS
4. **備份:** 重要的處理後資料可考慮其他備份方式

## 相關檔案

- `.gitignore` - Git 忽略檔案配置
- `data_download.py` - 資料集下載腳本
- `environment/requirements.txt` - Python 套件清單
- `README.md` - 專案說明和使用指南 