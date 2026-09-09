# 課程資料集下載

課程用到的 Kaggle 資料集不放進版控（`*.csv` 已被 `.gitignore` 排除），改用這裡的腳本下載。
**唯一的例外是二手車資料**，它已隨 repo 附上，見下方「不需要下載的資料」。

| 檔案 | 用途 |
| :--- | :--- |
| `data_download.py` | 主要下載工具。互動選單，可全部下載或挑單一資料集，失敗時自動改用 KaggleHub |
| `download_data.ipynb` | 同樣的事情但在 notebook 裡跑，方便逐格觀察下載結果 |

---

## 1. 設定 Kaggle 憑證（只做一次）

沒有憑證的話腳本會直接停在 `OSError: Could not find kaggle.json`。

1. 登入 Kaggle → 右上角頭像 → **Settings** → **API** → **Create New Token**，會下載 `kaggle.json`
2. 放到家目錄並收緊權限：

```bash
mkdir -p ~/.config/kaggle ~/.kaggle
cp ~/Downloads/kaggle.json ~/.config/kaggle/kaggle.json
cp ~/Downloads/kaggle.json ~/.kaggle/kaggle.json
chmod 600 ~/.config/kaggle/kaggle.json ~/.kaggle/kaggle.json
rm ~/Downloads/kaggle.json
```

**兩個位置都放**：新版 kaggle CLI（1.6+）讀 `~/.config/kaggle/`，
kagglehub 與舊版讀 `~/.kaggle/`，錯誤訊息只會提其中一個，兩邊都備著最省事。

Windows 放在 `C:\Users\<你的帳號>\.kaggle\kaggle.json`。

> **這是憑證，不要放進 repo。** `.gitignore` 已經擋掉 `kaggle.json`，
> 但更好的作法是根本不要複製到專案目錄裡。若不慎提交過，去 Kaggle 頁面
> **Expire Token** 後重新產一份。

> **清單裡沒有 competition 類資料集，這是刻意的。** Kaggle 的競賽 API 需要帳號具備競賽資格，
> 一般 API token 會收到 `401 Unauthenticated`——連列出競賽都不行，而且**接受條款也解決不了**。
> House Prices 與 Titanic 因此改用內容相同的 dataset 鏡像，檔名與 notebook 期望的一致。

---

## 2. 執行下載

```bash
# 在 repo 根目錄
uv run python data_mining_course/data_setup/data_download.py
```

從哪個目錄執行都可以——腳本以自己的檔案位置定錨，資料一律落在
`data_mining_course/datasets/raw/<folder>/`。

啟動後選單有三種模式：全部下載、依模組挑選、單一資料集。

---

## 3. 可下載的資料集

| 模組 | 資料集 | Kaggle ID | 落點 `datasets/raw/` |
| :--- | :--- | :--- | :--- |
| 模組三 | House Prices | `lespin/house-prices-dataset` | `house_prices/` |
| 模組四 | Titanic | `yasserh/titanic-dataset` | `titanic/` |
| 模組五 | Medical Cost Personal | `mirichoi0218/insurance` | `insurance/` |
| 模組六 | NYC Yellow Taxi Trip | `elemento/nyc-yellow-taxi-trip-data` | `nyc_taxi/` |
| 模組七 | Breast Cancer Wisconsin | `uciml/breast-cancer-wisconsin-data` | `breast_cancer/` |
| 模組八 | Electric Power Consumption | `uciml/electric-power-consumption-data-set` | `power_consumption/` |
| 模組八 | Hourly Energy Consumption (AEP) | `robikscube/hourly-energy-consumption` | `power_consumption/` |
| 模組九 | IMDB 50K Movie Reviews | `lakshmi25npathi/imdb-dataset-of-50k-movie-reviews` | `imdb_reviews/` |
| 模組九 | UrbanSound8K | `rupakroy/urban-sound-8k` | `urban_sound/` (6.7 GB) |
| 模組十 | Instacart Market Basket | `psparks/instacart-market-basket-analysis` | `instacart/` |
| 模組十 | Mall Customers | `vjchoudhary7/customer-segmentation-tutorial-in-python` | `mall_customers/` |
| 模組十 | Telco Customer Churn | `blastchar/telco-customer-churn` | `telco_churn/` |
| 專案 / 總整 | 100,000 UK Used Car | `adityadesai13/used-car-dataset-ford-and-mercedes` | 已附上，批次下載跳過 |

---

## 4. 不需要下載的資料

**二手車（`projects/project/car_data/`）已經在 repo 裡**，13 個 CSV、6.1 MB，
clone 完就能直接跑 `projects/project/car_market_eda.ipynb` 與 `capstone/capstone_used_car.ipynb`。

它是唯一破例進版控的資料集，理由有兩個：兩本 notebook 用寫死的相對路徑 `car_data/` 讀它，
而且課程的第一個完整專案不該要求學生先去申請 Kaggle API token。
資料授權為 **CC0-1.0（Public Domain）**，可自由散布；
來源為 Kaggle API `datasets metadata` 回報的 `licenses` 欄位。

**批次下載會自動跳過它**（選項 1「下載所有」與選項 2「依模組」都不含這一筆），
以免覆蓋掉版控裡的檔案。清單上仍看得到它並標著〔已隨 repo 附上〕，
是為了資料毀損時能用選項 3「下載單一資料集」重新取得——那時檔案會放回
`projects/project/car_data/`（而不是 `datasets/raw/`），因為 notebook 只認那個位置。

---

## 5. 出問題時

| 症狀 | 原因與處理 |
| :--- | :--- |
| `OSError: Could not find kaggle.json` | 憑證沒放好，回第 1 節 |
| `401 Unauthenticated` | 打到了競賽 API。清單裡已無 competition 類資料集，若你自行加了一筆，改找 dataset 鏡像 |
| `KeyError: 'username'` | `kaggle.json` 內容毀損，重新下載一份 token |
| 下載完 notebook 還是說找不到檔案 | 確認落點是 `data_mining_course/datasets/raw/`；`capstone/load_course_data.py` 也接受 repo 根目錄下的 `datasets/raw/`（舊版本的落點） |
| 網路斷在半路 | 直接重跑，已完成的資料集會被跳過 |

`capstone/load_course_data.py` 的所有 loader 在找不到檔案時都會退回可重現的合成資料並印出說明，
所以沒下載資料也能跑完整本 notebook，只是數字不是真實資料。

> **模組八要兩份電力資料。** `AEP_hourly.csv`（來自 hourly-energy-consumption）是五本 notebook 的主資料，
> `household_power_consumption.txt`（來自 electric-power-consumption-data-set）是其中四本的第二份示範資料。
> 兩者都落在 `power_consumption/`，缺了前者五本都只會跑降級的示意資料。

## 6. 不透過本腳本取得的資料

M09 的圖像與文字模態直接在 notebook 裡用 HuggingFace `load_dataset()` 抓
（`microsoft/cats_vs_dogs`、`stanfordnlp/imdb`），不需要 Kaggle 憑證，也不在上表。
清單裡曾有一筆 Dogs vs Cats 競賽資料，是改用 HuggingFace 之前的殘留，已移除。

磁碟空間提醒：全部下載約 14.5 GB，其中 `nyc_taxi` 6.9 GB、`urban_sound` 6.7 GB 就佔了絕大部分。
只上前八個模組的話這兩個可以不下載。
