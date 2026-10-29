# 舊版課程知識大綱：重設計參考基線

整理日期：2026-10-28。來源版本：`a9ef75a4c9fbb6aa6a6f791d66b06fffae053baa`。
目的：即使舊 notebook、腳本與投影片刪除，仍可知道原本教授哪些概念、技法、案例與判斷原則。
本檔自行包含摘要，不依賴舊教材檔案仍存在；來源路徑只是歷史定位，不作現行操作指令。

原版按章節／技法安排，共 11 個主模組、67 本主線 notebook，另有 M0 心法、二手車總整與專案。
`modules/extension/` 有部分重複教材，不重複計入 67 本。README 的摘要不能取代實際盤點。
原版並非全是傳統 ML：M09／M11 已有 Transformer、多模態與微調內容，以下獨立記錄。

## 1. 應保留的共同思考原則

來源：`data_mining_course/docs/course_charter.md`、`course_brief.md`、`M0_心法總綱.md`。

1. 先定義這次要做的決策，再選技術。
2. 用 SCQA 整理情境與問題，5 Why 尋找假說，KT 的 IS／IS NOT 界定範圍，SMART 篩選可衡量目標。
3. 尋找領域理由，確認資料是否真的量得到；沒有資料就記錄限制。
4. 先切再轉：所有學得的統計與參數只能來自訓練資料／訓練 Fold。
5. 特徵能講人話、資訊在預測時已可取得、拿掉後有可檢驗的影響。
6. 資料結構、標籤與張量形狀決定可接的模型；不能先挑模型再猜資料格式。
7. AI 可提假說、寫轉換；人要審問題、領域合理性、洩漏與結論。

重設計時的修正：5 Why 是提出因果假說的工具，不能代替因果識別或實驗證據。
舊程式的教學順序不一定等於正式驗證流程；保留概念不代表沿用全表補值／選特徵的做法。

## 2. M01：EDA 與資料理解（3 本）

歷史位置：`modules/module_01_eda_intro/`；主要案例 Titanic。

- Pandas：DataFrame／Series、讀取、欄位選取、過濾、基本聚合、`info`／`describe`。
- 單變數：目標比例、類別頻次、histogram、KDE、boxplot。
- 雙變數：scatter、類別對數值的 box／violin、類別對目標的分組比例。
- 多變數：hue、條件分組、Pearson 熱圖與交互關係。
- 系統化流程：問題 → schema／品質 → 單變數 → 雙變數 → 多變數 → 假說 → 處置。

保留的能力：知道為何畫圖、該按哪個群組比較，以及圖能支持什麼決定。
新 skill 對應：`tabular-diagnosis`。資料載入 API 操作可當先備知識，不再逐格抄寫。

## 3. M02：資料清理（4 本）

歷史位置：`modules/module_02_data_cleaning/`。

- 大檔 `chunksize` 分塊讀取、分塊統計與過濾。
- `.duplicated()`／`.drop_duplicates()`，全列與指定欄位重複。
- `astype`、`to_numeric(errors=...)`、`to_datetime`，型態與解析失敗。
- 字串去空白、大小寫、替換與欄位一致化。

保留的能力：定義 row identity、觀測粒度、schema、重複的業務含義。
新 skill 對應：診斷提出問題，`feature-engineering` 執行已核准的正規化；不在兩處重複清理。
分塊處理屬執行策略，不能每塊各自補中位數再宣稱等同全訓練集統計。

## 4. M03：缺失值與離群值（4 本）

歷史位置：`modules/module_03_missing_outliers/`；案例 House Prices。

- 缺失比例與 missingno 模式圖；MCAR／MAR／MNAR 的概念。
- 刪除、平均／中位數／眾數補值、SimpleImputer、KNNImputer。
- 視覺化、IQR、Z-score 與領域範圍判斷異常。
- 房價中的語意缺失、極值與欄位分策略處理。

保留的能力：區分「不存在」「未量測」「解析錯誤」，以及正常罕見事件與錯誤。
新 skill 對應：診斷產出候選處置，特徵工程在 Fold 內學補值與缺失指示。
缺失機制無法由缺失率直接證明；按全表 target 刪異常後評分不當成公平驗證。

## 5. M04：類別變數編碼（5 本）

歷史位置：`modules/module_04_categorical_encoding/`；案例 Titanic。

- Label／Ordinal 與 One-Hot，名目與序位類別的差異。
- Count／Frequency encoding，類別出現次數與相對頻率。
- Target encoding、naive 洩漏、K-Fold cross-fitting、smoothing。
- 高基數：罕見類別、hashing 與領域合併。
- Titanic 艙等等欄位的編碼選擇。

保留的能力：選擇表示方式，處理未知類別，避免引入虛假的大小關係。
新 skill 對應：`feature-engineering`；計數也只能從訓練 Fold 學得，target encoding 為進階功能。

## 6. M05：縮放與分佈轉換（4 本）

歷史位置：`modules/module_05_scaling_transformation/`；案例 Insurance。

- StandardScaler、MinMaxScaler、RobustScaler 與離群值影響。
- log／log1p、Box-Cox、Yeo-Johnson、Q-Q 圖與偏態。
- 比較轉換前後的分佈與尺度。
- 區分輸入 X 轉換與目標 y 轉換。

保留的能力：依模型與欄位範圍選轉換；樹模型一般不因量綱需要標準化。
新 skill 對應：`feature-engineering`＋`model-evaluation`。
轉換 y 後須回原業務尺度評估；分佈更像常態不等於模型一定更好。

## 7. M06：特徵創造（4 本）

歷史位置：`modules/module_06_feature_creation/`；案例 NYC Taxi。

- 交互項、PolynomialFeatures、具領域理由的手工組合。
- groupby 多欄聚合、群組統計合併回原列。
- 日期拆解、時間差、週期資訊、歷史聚合。
- Taxi 的時間與空間特徵；行程時長推導與長尾轉換。

保留的能力：為每個特徵寫公式、單位、假說與資訊可用時間。
新 skill 對應：`feature-engineering`；多項式保留概念介紹，不預設暴力生成。
Taxi 的 dropoff 可用來建 target，不能拿來預測起程時未知的時長。

## 8. M07：特徵選擇與降維（5 本）

歷史位置：`modules/module_07_feature_selection/`；案例 Breast Cancer。

- Filter：相關性、ANOVA F、chi-square、SelectKBest。
- Wrapper：前向／後向選擇、SequentialFeatureSelector、RFE／RFECV。
- Embedded：L1／Lasso、樹重要性、SelectFromModel。
- PCA 與變異保留；t-SNE 的探索用途。
- 比較原始特徵、選擇器與降維後的分類表現。

保留的能力：在性能、冗餘、成本與可解釋性間取捨，辨認選擇偏差。
新 skill 對應：`model-evaluation` 管 Fold 內篩選與消融；`model-explanation` 解讀排名。
不能把 t-SNE 圖上群聚直接當泛化驗證，也不能把 PCA 成分當原始業務因子。

## 9. M08：時間序列特徵（5 本）

歷史位置：`modules/module_08_time_series/`；案例 AEP 小時負載及 UCI household 電力。

- lag、shift、歷史移動平均／標準差／最大最小值、rolling／expanding。
- hour／weekday／month、假日與週期 sin／cos。
- 加法／乘法季節分解、趨勢與週期。
- 時序切分、TimeSeriesSplit、LightGBM 回測。

保留的能力：定義發報時點、預測 horizon、可見歷史、缺時、採樣粒度與標籤延遲。
新 skill 對應：特徵工程的時序模式＋模型評估的時間切分模式，不另造第二套工作流。
兩份電力資料單位與粒度不同；雙向分解或 centered rolling 不自動成為合法預測特徵。

## 10. M10：傳統資料探勘與端到端模型（9 本）

歷史位置：`modules/module_10_data_mining_applications/`。

- 關聯規則（2 本）：交易 One-Hot、Apriori、support／confidence／lift；Instacart。
- 分群（3 本）：K-Means、肘部法、silhouette、DBSCAN 的 eps／min_samples；Mall Customers。
- 樹模型（3 本）：XGBoost／LightGBM、重要性、分類指標；Telco Churn。
- 端到端（1 本）：ColumnTransformer、Pipeline、資料切分、前處理與模型整合。

保留的能力：區分分類、回歸、無監督分群與關聯探索，不混用驗收指標。
新 skill 對應：核心為 `model-evaluation`；分群與關聯規則列延伸功能，待有課程需求再封裝。
關聯規則不等於推薦系統全貌，silhouette 不等於業務分群有效，重要性不等於因果。

## 11. 總整與專案

歷史位置：`capstone/`、`projects/project/`、`projects/midterm/`、`projects/final/`。

- UK Used Cars：問題定義、5 Why、SMART、業務理由、特徵、先切再轉與口頭辯護。
- 汽車市場 EDA／視覺化：品牌與車型分群、規格與價格的多角度比較。
- 隨庫品牌 CSV 與 unclean 範例：可離線練習品質診斷；不同子集不可不加區分地合併。
- 期中／期末：以資料分析、特徵工程、成果說明驗收綜合能力。

保留的能力：同一決策問題走完一輪，報告說得出限制與取捨。
案例資料與例程是兩種資產；移除 notebook 不代表應連帶刪掉合法可用的資料。
未來資料來源改由案例契約記錄，不要求保留舊目錄才能運作。

## 12. M09／M11：另存的現代多模態內容

M09 共 18 本：文字 5、影像 5、音訊 4、影片 3、圖文配對 1。

- 經典表示：BoW／TF-IDF／靜態詞向量、色彩直方圖／HOG、STFT／MFCC。
- 文字：Subword tokenizer、padding／truncation、上下文與句向量、JSONL／chat template／packing。
- 影像：tensor、transforms、訓練／驗證增強、ViT／CLIP、Zero-shot。
- 音訊：重採樣、波形／log-mel、Whisper／wav2vec2。
- 影片：解碼、frame／clip 抽樣、VideoMAE；圖文配對與 VLM 格式。
- 案例：IMDB、Dogs vs Cats、UrbanSound、真實影片與圖文資料。

M11 共 6 本：資料到模型地圖、文字下游、影像下游、音訊下游、影片下游、生成／多模態藍圖。
含 DistilBERT／ViT 微調、Whisper ASR、wav2vec2／VideoMAE、LoRA／QLoRA、SFT、RAG 與 Diffusion／VLM 藍圖。

這條線保留知識清單，但不列第一版表格特徵工作坊的必備 skill。
若日後重建，按資料模態與訓練目標另行設計，不能全部硬塞入表格模型歸因。

## 13. 原有與新增能力的界線

原有重點是技法理解、處理流程與模型重要性；新工作坊新增可維運的封裝與交付邊界：

- 特徵假說登錄與人類採納／否決紀錄。
- 相同 Fold、相同資料列的配對比較與消融。
- OOF 歸因、輸出尺度、跨 Fold 穩定性與局部錯誤案例。
- 一次 Holdout 驗收、不可變產物與完整來源追溯。
- 面向決策者的證據報告，而非只有模型分數或圖表集合。

上述新增項目是重設計要求，不聲稱原版 67 本已完整實作。

## 14. 刪除舊程式後如何回查

本次只整理，不執行刪除。若日後要回查來源，可用來源 commit 讀取而不恢復整棵目錄：

```bash
git show a9ef75a4c9fbb6aa6a6f791d66b06fffae053baa:data_mining_course/docs/course_brief.md
git ls-tree -r --name-only a9ef75a4c9fbb6aa6a6f791d66b06fffae053baa -- data_mining_course/modules
```

Git 歷史只保留已追蹤的檔案；未追蹤的私人資料與本機產物不能靠該 commit 回復。
這份大綱應留在新的 `docs/`，不放進準備移除的舊教材目錄。
