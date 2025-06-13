# 🎯 資料探勘與特徵工程教學課程

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://python.org)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Course Status](https://img.shields.io/badge/status-Active-brightgreen.svg)](README.md)

> 一套完整的資料探勘與特徵工程實戰課程，從基礎 EDA 到進階多模態特徵工程，涵蓋 10 個核心模組與 54 個實戰案例。

## 📋 目錄

- [🚀 快速開始](#-快速開始)
- [📚 課程內容](#-課程內容)
- [🛠️ 環境設置](#-環境設置)
- [📊 資料集管理](#-資料集管理)
- [🎯 學習路徑](#-學習路徑)
- [💡 特色功能](#-特色功能)
- [🤝 貢獻指南](#-貢獻指南)
- [📝 許可證](#-許可證)

## 🚀 快速開始

### 1️⃣ 克隆專案
```bash
git clone https://github.com/你的用戶名/iSpan_python-FE_DM-cookbooks.git
cd iSpan_python-FE_DM-cookbooks
```

### 2️⃣ 設置環境
```bash
# 創建虛擬環境
python -m venv venv
source venv/bin/activate  # Linux/Mac
# 或
venv\Scripts\activate     # Windows

# 安裝依賴
pip install -r data_mining_course/environment/requirements.txt
```

### 3️⃣ 配置 Kaggle API
```bash
# 下載 kaggle.json 從 https://www.kaggle.com/account
mkdir ~/.kaggle
cp kaggle.json ~/.kaggle/
chmod 600 ~/.kaggle/kaggle.json
```

### 4️⃣ 下載資料集
```bash
python data_download.py
```

### 5️⃣ 開始學習
```bash
cd data_mining_course
jupyter notebook
```

## 📚 課程內容

### 📊 課程概覽

| 模組 | 主題 | 筆記本數 | 狀態 | 主要技術 |
|:---:|:---|:---:|:---:|:---|
| **M01** | 課程導入與EDA複習 | 3 | ✅ | Pandas, Matplotlib, Seaborn |
| **M02** | 資料清理與預處理 | 4 | ✅ | 分塊處理, 重複值, 型態轉換 |
| **M03** | 缺失值與異常值處理 | 4 | ✅ | 插補方法, 異常檢測 |
| **M04** | 類別變數編碼 | 5 | ✅ | One-Hot, Label, Target Encoding |
| **M05** | 特徵縮放與變數轉換 | 4 | ✅ | StandardScaler, 冪轉換 |
| **M06** | 特徵創造 | 4 | ✅ | 交互特徵, 聚合特徵 |
| **M07** | 特徵選擇與降維 | 5 | ✅ | 過濾法, 包裹法, PCA |
| **M08** | 時間序列特徵工程 | 5 | ✅ | 滯後特徵, 滑動窗口 |
| **M09** | 多模態特徵工程 | 11 | ✅ | 文字, 圖像, 音訊 |
| **M10** | 資料探勘應用 | 9 | ✅ | 關聯規則, 聚類, 樹模型 |

**總計**: 54 個實戰筆記本 | 12 個 Kaggle 資料集 | 10+ 小時教學內容

### 🎯 核心學習主題

#### 🔍 **基礎數據分析**
- **探索性數據分析 (EDA)**: 統計描述、分佈視覺化、相關性分析
- **數據清理**: 重複值處理、異常值檢測、數據一致性檢查
- **數據預處理**: 型態轉換、編碼轉換、格式標準化

#### 🛠️ **特徵工程技術**
- **缺失值處理**: 簡單插補、多重插補、進階插補策略
- **類別編碼**: One-Hot、Label、Ordinal、Target Encoding
- **數值特徵**: 標準化、正規化、分箱、冪轉換
- **特徵創造**: 多項式特徵、交互特徵、聚合統計特徵

#### 🚀 **進階技術**
- **特徵選擇**: 過濾法、包裹法、嵌入法、遞歸特徵消除
- **降維技術**: PCA、t-SNE、UMAP
- **時間序列**: 滯後特徵、滑動統計、季節性分解
- **多模態特徵**: 文字 (TF-IDF, Word2Vec)、圖像 (CNN, HOG)、音訊 (MFCC)

#### 🎲 **資料探勘應用**
- **關聯規則挖掘**: Apriori 演算法、購物籃分析
- **聚類分析**: K-Means、DBSCAN、層次聚類
- **樹模型特徵重要性**: XGBoost、LightGBM、Random Forest

## 🛠️ 環境設置

### 系統需求
- **Python**: 3.8+ 
- **記憶體**: 8GB+ (推薦 16GB)
- **硬碟空間**: 10GB+ (含資料集)
- **Kaggle 帳號**: 用於資料集下載

### 主要依賴套件
```
pandas>=1.3.0
numpy>=1.21.0
matplotlib>=3.3.0
seaborn>=0.11.0
scikit-learn>=1.0.0
xgboost>=1.5.0
lightgbm>=3.3.0
jupyter>=1.0.0
kaggle>=1.5.0
```

### 🐳 Docker 支援
```bash
cd data_mining_course/environment/docker
docker-compose up -d
```

## 📊 資料集管理

### 🎯 智能下載系統
本專案採用 **零資料集上傳** 策略，所有資料集通過 `data_download.py` 自動下載：

```bash
python data_download.py
```

#### 互動式選單
- 📥 **下載所有資料集** (12個 Kaggle 資料集)
- 📂 **按模組下載** (選擇特定學習模組)
- 🎯 **單一資料集下載** (精確選擇)

### 📋 完整資料集清單

| 資料集 | 大小 | 用途 | 來源 |
|:---|:---:|:---|:---:|
| House Prices | ~80MB | 回歸、缺失值處理 | Kaggle Competition |
| Titanic | ~60KB | 分類、特徵工程 | Kaggle Competition |
| Insurance | ~50KB | 回歸、數值特徵 | Kaggle Dataset |
| NYC Taxi | ~1.6GB | 時空特徵、大數據 | Kaggle Competition |
| Breast Cancer | ~120KB | 分類、特徵選擇 | UCI ML Repository |
| Power Consumption | ~2.1MB | 時間序列 | UCI ML Repository |
| IMDB Reviews | ~66MB | 文字分析、NLP | Kaggle Dataset |
| Dogs vs Cats | ~800MB | 圖像分類、CNN | Kaggle Competition |
| UrbanSound8K | ~6GB | 音訊分析、頻譜 | Kaggle Dataset |
| Instacart | ~1.4GB | 關聯規則、推薦 | Kaggle Competition |
| Mall Customers | ~4KB | 聚類分析 | Kaggle Dataset |
| Telco Churn | ~950KB | 分類、商業分析 | Kaggle Dataset |

## 🎯 學習路徑

### 🎓 **初學者路徑** (2-3 週)
```mermaid
graph LR
    A[M01: EDA基礎] --> B[M02: 數據清理]
    B --> C[M03: 缺失值處理]
    C --> D[M04: 類別編碼]
    D --> E[實戰項目: Titanic]
```

### 🚀 **進階路徑** (4-6 週)
```mermaid
graph LR
    A[M05: 特徵縮放] --> B[M06: 特徵創造]
    B --> C[M07: 特徵選擇]
    C --> D[M08: 時間序列]
    D --> E[實戰項目: House Prices]
```

### 🎯 **專家路徑** (6-8 週)
```mermaid
graph LR
    A[M09: 多模態特徵] --> B[文字特徵工程]
    B --> C[圖像特徵提取]
    C --> D[音訊信號處理]
    D --> E[M10: 端到端流程]
```

## 💡 特色功能

### 🎨 **視覺化工具**
- 互動式 EDA 儀表板
- 特徵重要性可視化
- 模型效能比較圖表
- 特徵分佈動態圖

### ⚡ **自動化工具**
- 一鍵資料集下載
- 自動特徵工程流水線
- 模型調參助手
- 實驗結果追蹤

### 📚 **學習輔助**
- 詳細註解的程式碼
- 理論與實作結合
- 常見錯誤解析
- 最佳實踐指南

## 🏗️ 專案結構

```
📁 iSpan_python-FE_DM-cookbooks/
├── 📂 data_mining_course/           # 主課程目錄
│   ├── 📂 modules/                  # 教學模組
│   │   ├── 📂 module_01_eda_intro/
│   │   ├── 📂 module_02_data_cleaning/
│   │   └── ...
│   ├── 📂 datasets/                 # 資料集目錄
│   │   ├── 📂 raw/                  # 原始資料
│   │   └── 📂 processed/            # 處理後資料
│   ├── 📂 utils/                    # 工具函數
│   ├── 📂 templates/                # 範本檔案
│   └── 📂 environment/              # 環境配置
├── 📂 course_slides/                # 課程簡報
├── 📄 data_download.py              # 資料下載腳本
├── 📄 .gitignore                    # Git 忽略檔案
└── 📄 README.md                     # 專案說明
```

## 🤝 貢獻指南

我們歡迎各種形式的貢獻！

### 🎯 貢獻方式
- 🐛 **Bug 回報**: 發現問題請開 Issue
- 💡 **功能建議**: 提出新想法或改進建議
- 📝 **文檔改進**: 完善說明或添加註解
- 🔧 **程式碼貢獻**: 提交 Pull Request

### 📋 貢獻流程
1. Fork 本專案
2. 創建功能分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送到分支 (`git push origin feature/AmazingFeature`)
5. 開啟 Pull Request

## 📞 聯絡資訊

- **課程討論**: [開啟 Issue](../../issues)
- **技術問題**: [查看 FAQ](data_mining_course/docs/faq.md)
- **課程大綱**: [查看詳細說明](data_mining_course/docs/syllabus.md)

## 📝 許可證

本專案採用 MIT 許可證。詳見 [LICENSE](LICENSE) 檔案。

---

### 🌟 如果這個專案對你有幫助，請給我們一個 Star！

[![GitHub stars](https://img.shields.io/github/stars/你的用戶名/iSpan_python-FE_DM-cookbooks.svg?style=social&label=Star)](https://github.com/你的用戶名/iSpan_python-FE_DM-cookbooks) 