# 課程資料集下載

課程用到的 Kaggle 資料集不放進版控（`*.csv` 已被 `.gitignore` 排除），改用這裡的腳本下載。
**唯一的例外是二手車資料**，它已隨 repo 附上，見下方「不需要下載的資料」。

| 檔案 | 用途 |
| :--- | :--- |
| `data_download.py` | 主要下載工具。互動選單，可全部下載或挑單一資料集，失敗時自動改用 KaggleHub |
| `download_data.ipynb` | 同樣的事情但在 notebook 裡跑，方便逐格觀察下載結果 |
| `export_hf_datasets.py` | 把 HuggingFace 那兩份資料落地成 `datasets/raw/` 的一般檔案（見第 6 節） |

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

## 6. HuggingFace 那兩份資料

M09 與 M11 有五本 notebook 從 HuggingFace 取資料，不經過 Kaggle：

| 資料集 | 用它的 notebook |
| :--- | :--- |
| `microsoft/cats_vs_dogs` | M09 `02_image_features/05_dogs_cats_case`、M11 `03_image_downstream` |
| `stanfordnlp/imdb` | M09 `01_text_features/05_imdb_case`、M11 `02_text_downstream`、extension `02_text_downstream` |

不處理的話它們只會存在每個人自己的 `~/.cache/huggingface`——上課時等於每個學生現場各下載一次
（貓狗約 700 MB），教室網路一塞整堂就停擺，而且老師沒辦法事先備好發下去。所以先落地：

```bash
uv run python data_mining_course/data_setup/export_hf_datasets.py
```

產生：

```
datasets/raw/imdb_hf/{train,test}.csv        欄位 text, label
datasets/raw/dogs_vs_cats/{cat,dog}/*.jpg    imagefolder 版面，cat=0, dog=1
```

**五本 notebook 都改成優先讀這裡**，找不到才回頭抓 HuggingFace。所以老師跑一次匯出、
把 `datasets/` 發下去，學生現場零下載。落地檔與 HF 的欄位完全一致
（`image`/`labels`、`text`/`label`），notebook 其餘程式碼不用改。

只要小樣本先試：

```bash
uv run python data_mining_course/data_setup/export_hf_datasets.py --per-class 200
uv run python data_mining_course/data_setup/export_hf_datasets.py --only imdb
```

> 清單裡曾有一筆 Dogs vs Cats 競賽資料，是改用 HuggingFace 之前的殘留，已移除。
> `~/.cache/huggingface` 裡的 `ag_news`、`cifar10`、`librispeech_dummy` 同樣沒有任何 notebook 引用。

---

## 7. 資料一律留在課程樹下，不散到家目錄

兩支腳本都把下載暫存指回 `data_mining_course/datasets/` 底下，而不是預設的 `~/.cache`：

| 工具 | 環境變數 | 落點 |
| :--- | :--- | :--- |
| HuggingFace `datasets` | `HF_HOME` | `datasets/.hf_cache/` |
| KaggleHub（CLI 失敗時的後備） | `KAGGLEHUB_CACHE` | `datasets/.kagglehub_cache/` |

Kaggle CLI 本身直接寫進 `--path`，不經過快取。兩個暫存目錄都已被 `.gitignore` 擋住，
**匯出完成後可以整個刪掉**，notebook 只讀 `raw/`。

已經有既有快取、不想重抓的人可以自己覆寫，例如：

```bash
HF_HOME=~/.cache/huggingface uv run python data_mining_course/data_setup/export_hf_datasets.py
```

> 模型權重（DistilBERT、ViT 等）不在此列，仍走 `~/.cache/huggingface/hub`。
> 那是 `transformers` 的行為，與資料無關；要一起收進課程樹得另外設 `HF_HUB_CACHE`。

磁碟空間提醒：Kaggle 全下載約 14.5 GB，其中 `nyc_taxi` 6.9 GB、`urban_sound` 6.7 GB 佔絕大部分；
HF 落地再加約 830 MB。只上前八個模組的話那兩個大的可以不下載。
