# AI 協作特徵工程與模型決策透明化工作坊：整體設計

> 前版提案。使用者已改採按功能封裝 Skill，並計畫移除舊例程。封裝、路徑與程式保留策略以 [新版 Skill 架構](2026-10-28-skill-based-workshop-design.md) 為準；本檔保留案例、驗證與教學決策的設計脈絡，不作新架構實作入口。

日期：2026-10-28。狀態：架構提案，供審查；本次交付為設計，尚未實作或訓練。
分支：`feature/ai-feature-decision-workshop`，起點 `a9ef75a`。

## 1. 核心判斷與定位

值得做。把本庫既有的「技法 notebook」串成可重現、可質疑、可拒絕的業務決策流程，才是突破。
學員負責決策問題、資訊可用時間、假說與驗收；AI 負責寫程式、計算與可追溯的證據整理。
沿用既有 `course_charter.md` 的 5 Why、SMART、先切再轉與領域理由，不另造方法論。

「白盒化」是教學名稱：樹模型仍是複雜模型，SHAP 提供預測歸因，不會把它變成因果模型。
交付必須分開寫觀察、模型關聯、領域假說與待驗證行動。SHAP 不提供介入效果的保證。

預設學員已會 Python／基本 ML；12 小時實作、6 小時精簡、18 小時完整三案例。
課時是設計者暫定，可調整；現階段不依賴學員先取得 Kaggle 憑證。

### 目標

- 完成一題從資料字典、EDA、特徵審查、同 Fold 對照到 SHAP 決策報告。
- 能找出 AI 提案中的洩漏、錯誤單位、錯誤驗證與無證據的因果敘述。
- 每個結論可追到資料版本、切分、程式、指標或圖表。
- 產生可獨立執行的腳本；notebook 保留作為教學工具。

### 非目標與反目標

不建多 Agent 調度、AutoML 平台、資料庫、Web 服務或 MLOps 服務。
不強求新特徵提升，不為湊 10 個特徵而暴力交叉，不把臨界點包裝成現場安全標準。
本次不改寫 M01–M11、不更新 lockfile、不下載大資料、不執行全庫 notebook。

## 2. 已有資產與整合選擇

以下以實際程式／notebook 為依據，不能只依根 README 的摘要數字。

| 既有資產 | 重用方式 | 需補上的決策能力 |
|---|---|---|
| M01 系統化 EDA 指南與視覺化 | 品質 → 單變數 → 雙變數 → 多變數 → 假說 | 將每張圖連到一個處置／特徵假說 |
| M02、M03 清理、缺失與離群 | 型態、重複、語意缺失、極值診斷 | 統計量僅在訓練 Fold 學習；極值不自動刪除 |
| M04、M05 編碼、縮放與轉換 | One-Hot、罕見類別、RobustScaler、log1p | 比較線性與樹模型需求；轉換目標後回原尺度評估 |
| M06 交叉與聚合 | 具領域理由的 ratio、交互與群組統計 | 學得的群組統計留在 Fold；禁用目標欄造特徵 |
| M07 選擇與乳癌案例 | Filter、Embedded、RFE、PCA 對照 | 以消融與跨 Fold 穩定性決定去留 |
| M08 AEP 電力與 lag／rolling | 日曆、歷史視窗、TimeSeriesSplit | 固定預測時點、標籤時間與可見歷史 |
| M10 樹模型、Telco、Pipeline | LightGBM／XGBoost、編碼、指標 | 同 Fold 公平比較與 OOF 解釋 |
| 二手車總整與本地品牌 CSV | 首個可離線啟動的端到端主線 | 多模型對照、特徵登錄與證據報告 |
| M09、M11 多模態與下游訓練 | 作為其他課程路徑保留 | 不混入本工作坊核心 |

專案已使用 `uv`、`pyproject.toml`、`uv.lock`，Python 範圍為 >=3.9,<3.13。
沿用現況，不為此設計換 Poetry 或升級 Python。LightGBM、XGBoost 已列核心依賴；
SHAP 尚未列入 pyproject／lockfile，後續採獨立 `xai` extra 並驗證相容性，不聲稱已可執行。
目前沒有看到本工作坊可直接沿用的測試套件；未來補 `pytest` 開發依賴與高價值邊界測試。

不直接重用 `utils/preprocessing.py` 的全表補值，也不把 capstone loader 的合成資料後備當真實案例。
現有 notebook 可當技法參考，不代表其全表前處理可直接搬入正式驗證。

## 3. 案例組合

| 層級／題目 | 決策問題與目標 | 為何選它 | 主限制 |
|---|---|---|---|
| 主線：UK Used Cars | 依上架可用屬性估計刊登價格 `price`，供人工定價比較 | 現有總整、真實 CSV 隨庫，立即可做 | 刊登價不等於成交價；沒有可信上架時間或車輛 ID |
| 數值分類：Breast Cancer Wisconsin Diagnostic | 以細胞核量測分類 `diagnosis`，M=1 | 現有 M07、30 數值特徵、共線性與歸因教學 | 不是一般血液檢驗，也不是臨床部署驗證 |
| 時序：AEP Hourly Energy Consumption | 在 t 時刻預測 t+1 小時 `AEP_MW` | 現有 M08、真正 lag／rolling 與回測 | 是區域電力負載，沒有設備故障標籤 |
| 選修：House Prices | `SalePrice` 回歸 | 語意缺失、長尾、共線性 | 不能依全表目標刪極值後再宣稱無偏驗證 |
| 選修：Insurance | `charges` 費用回歸 | 偏態、交互、原尺度誤差 | 費用模型不等於疾病因果 |
| 選修：Telco Churn | `Churn` 分類與留客名單排序 | 現有 M10、混合型欄位與 PR 曲線 | 橫斷面資料無法證明促銷介入可降低流失 |
| 選修：Titanic／NYC Taxi | 類別編碼／起程時行程時間預測 | 現有 M04／M06 | Taxi 必須先確認檔案 schema；dropoff 時間、總費用不可作起程時特徵 |

不硬把現有資料改稱「軸承退化」或「血液臨床指標」；未來新增感測故障資料是獨立案例。
Instacart、Mall Customers、IMDB、音訊與影像不塞進這套監督式表格 SHAP 流程。
完整案例欄位、EDA、特徵與模型規劃見 [案例與教材對照](ai-workshop-cases.md)。

## 4. 三種架構方案與決策

1. 原地重寫全部 notebook：牽連既有教學順序與讀檔路徑，回歸成本高。
2. 新建一個完全獨立 repo：重複維護資料與依賴，失去舊教材導流。
3. 在本庫新增 `workshop/` 教學入口、共用既有原始資料與依賴：採用。

理由：一套小型腳本即可服務三題；只把確實共用的切分、驗證與報告邏輯共用。
不建 plugin registry、抽象 Agent base class 或多份相同規則。分案例只保留欄位與純轉換函式。

## 5. 目標目錄與資料流

下列是後續實作的目標結構，不代表這些檔案已建立。四階段規範的設計稿在配套文件。

```text
workshop/
├── README.md                         # 教學入口、環境與四階段 runbook
├── AGENTS.md                         # 工作坊邊界、共用規範入口
├── CLAUDE.md                         # 薄入口，引用同一套規範
├── .claude/
│   ├── 01_eda.md
│   ├── 02_feature_eng.md
│   ├── 03_modeling.md
│   └── 04_interpretability.md
├── cases/
│   ├── used_cars/                     # 各案例同樣四份 Markdown
│   │   ├── data_contract.md           # 來源、欄位、目標、可用時間、切分
│   │   ├── feature_registry.md        # 假說、公式、狀態、審核記錄
│   │   ├── experiment_plan.md         # 指標、模型、預算與凍結選擇
│   │   └── decisions.md               # 人類判斷、否決理由與證據
│   ├── breast_cancer/
│   └── power_consumption/
├── scripts/
│   ├── validate_data.py               # 唯讀 schema／來源／品質檢查
│   ├── run_eda.py                     # 僅訓練開發集的探索性分析
│   ├── build_features.py              # 無全表 fit 的特徵匯出
│   ├── train_eval.py                  # 固定切分、Pipeline、OOF、消融
│   └── explain_report.py              # OOF SHAP 與可追溯報告
├── lib/
│   ├── data.py                       # 明確路徑、schema、穩定列 ID
│   ├── features.py                   # 三案例純衍生與 Fold 內轉換
│   ├── validation.py                 # 建立與重用切分 manifest
│   ├── modeling.py                   # 管線、訓練與統一指標
│   └── reporting.py                  # 圖表、歸因與報告資料整理
├── templates/final_decision_report.md
├── tests/                            # 人工小資料，無 Kaggle 認證需求
└── workspace/                        # 每次運算新建 run，不進版控
    └── <case>/<run_id>/
        ├── processed/                # 特徵、字典；尚未 fit 的矩陣
        ├── artifacts/                # 每 Fold 前處理器／模型，僅本機
        └── output/                   # manifest、指標、OOF、圖與報告
```

```mermaid
flowchart LR
  A[人類：問題／欄位可用時間] --> B[原始資料唯讀驗證]
  B --> C[固定 Holdout 與開發集 Fold]
  C --> D[開發集 EDA 與假說審核]
  D --> E[純衍生特徵]
  E --> F[每 Fold fit 前處理／模型]
  F --> G[OOF 指標／歸因／消融]
  G --> H[人類凍結方案]
  H --> I[一次 Holdout 驗收]
  I --> J[證據報告與人類決策]
```

原始資料只有一個來源：二手車讀既有 `car_data/`，其他讀 `data_mining_course/datasets/raw/`。
不另複製成 `workshop/data/raw/dataset.csv`，也不以符號連結隱藏來源。
所有路徑相對 repo root 定錨，拒絕依執行目錄多路猜檔。缺檔就停並提供下載指引，不自動改用合成資料。
`run_id` 使用時間與短隨機碼；存在就拒絕覆寫。產物寫到新 run，失敗保留記錄。
每個階段寫獨立且不可變的檔案：CV 寫 `metrics_cv.json`，finalize 才建立含 Holdout 的
`metrics.json`；初始 manifest 不回寫，後續各階段另寫 stage manifest。

### 模組邊界

`scripts → lib`；`lib` 不讀 notebook、不 import capstone、不在 import 時訓練或下載。
資料層只讀原檔；特徵層不讀 Holdout 標籤；模型層不撰寫業務結論；報告層不重新訓練。
命令列只負責參數與協調，不複製計算邏輯。先用函式與少量 dataclass，無通用工作流引擎。

## 6. 驗證與比較契約

### 先切再分析、再學習

- 固定 seed=42；二手車與乳癌先保留 20% Holdout；電力保留尾端連續 20%。
- 全體僅做預先定義的 schema、hash、重複與時間索引檢查；有標籤的 EDA 僅限開發集。
- 二手車：跨品牌先正規化 `model` 空白與 `tax(£)`，只讀九品牌主檔。
  不再併入 focus／cclass 與 unclean 檔，避免重複來源。固定規則去重後建立切分。
  無車輛 ID 時，將正規化後所有非目標原始屬性的相同列組成 proxy group；同組不得跨集合。
  以 GroupShuffleSplit 保留約 20% 群組後，在開發集採 GroupKFold(5)。
  proxy group 不代表真正車輛身份，仍須揭露識別限制；不同價格的同屬性列不擅自刪除。
- 乳癌：stratified holdout + StratifiedKFold(5, shuffle=True, random_state=42)。
  若存在重複病人／量測群組則改 StratifiedGroupKFold，先檢查每 Fold 類別是否足夠。
- 電力：按真實時間排序，開發集 TimeSeriesSplit(5)；先依標籤可取得時間排除跨界樣本。
  固定一小時步進，不能把製造年份或原始 CSV 列順序當收集時間。
- 缺失補值、縮放、分位截尾、類別詞彙、頻率／群組統計、選擇器全部只 fit 訓練 Fold。
  同樣的轉換器 transform 驗證集；未知類別使用明確後備策略。
- 目標編碼不列入核心，選修才實作訓練 Fold 內 cross-fitting。
- CV 外層驗證集不供 early stopping 或搜尋選項；預設固定樹數／參數。
  如加調參、early stopping、校準、分類門檻，必須用外層訓練集內的 inner split。

### 公平比較

每題先建立簡單基準與 LightGBM；XGBoost 是有時數才跑的 challenger。
依序比較 Dummy／季節基準、原始特徵簡單模型、原始特徵 LightGBM、衍生特徵 LightGBM。
衍生前後保持相同列、Fold、模型參數與預算；另列同特徵下的模型差異，不能把換模型的提升算給特徵工程。

每 Fold 保留配對分數、原始方向的 `variant - baseline`、以及「正值代表較好」的 improvement。
例如 MAE improvement = baseline MAE - variant MAE；ROC-AUC improvement 反向。
報告 Fold 平均、標準差與逐 Fold 方向；5 Fold 標準差不當成獨立抽樣的信賴區間。
時序顯示每段表現與跨度，不只報平均。

Gain／Split 只是篩查，不能因值近零直接刪欄。Permutation 重要性與移除後重訓共同判斷。
篩選方案要在 inner CV 內選擇並在外層評估；課時不足時只提汰除候選，不能把探索後的分數當無偏提升。
同一開發集反覆比較具有選擇偏差，最終方案凍結後才看 Holdout 一次。
若 Holdout 不理想，如實報告；後續調整列為新實驗，不再把該 Holdout 當未見資料。
G3 後由 `train_eval.py --stage finalize` 讀取人類凍結決定，以全開發集重新 fit 所選
Pipeline，儲存 `artifacts/final_pipeline` 並一次評估 Holdout。不使用 Fold ensemble 作最終模型。
報告層只讀已存在的模型與結果。OOF 歸因解釋的是各 Fold 模型；不得冒稱最終重訓模型的逐筆歸因。

## 7. 解釋性與門檻邊界

- 原生可解釋對照：標準化 Logistic／Ridge 係數與可選深度 3 決策樹；保留單位與共線性限制。
- 若最終選中簡單模型，必要交付改為係數／規則圖及局部預測分解。LightGBM SHAP 仍可作
  比較模型教學圖，必須具名標示為 challenger，不能當最終模型的解釋或硬性必交品。
- 複雜樹模型：每 Fold 使用該 Fold 已 fit 的轉換器及模型，對其外層驗證列計算 Tree SHAP。
  背景樣本僅取該 Fold 訓練資料，固定 seed，預設上限 100；解釋樣本每 Fold 上限 200。
- 記錄 `model_output`、`feature_perturbation`、類別、expected value、樣本與套件版本。
  raw output 不一律叫 log-odds；以模型實際輸出為準，回歸可能是原尺度或 log1p 尺度。
- 使用前驗證 SHAP 維度、類別索引，以及 expected value + SHAP 和重建模型輸出。
  乳癌正類統一為惡性 M=1，禁止沿用 sklearn 內建標籤方向而不轉換。
- Fold 類別詞彙可能不同：先在 Fold 內計算，再按具名特徵對齊；聚合 One-Hot 回原始欄位。
  不把衍生特徵再任意回分給父欄位。報告展示抽樣涵蓋度及跨 Fold 排名變化。
- 至少 Summary、前二名 Dependence、兩筆局部解釋（失誤與典型）；相關性高時提示歸因可互相分攤。
- Dependence 的轉折只能稱「候選區間」，記錄分箱樣本數、跨 Fold 是否一致；沒有穩定轉折就寫沒有。
  預設不自動挑選最漂亮的門檻；領域專家提出候選，開發集驗證，再交人類審查。
- 分類機率門檻是另外一件事，依漏判／誤報成本在 inner validation 選擇。
  因果、臨床或現場控制建議須另有外部驗證；本課交付待驗證建議。

## 8. 人機協作與執行權限

共用規範只有一份，放 `.claude/`；Codex 透過 `AGENTS.md` 與明確任務引用載入，
Claude 透過 `CLAUDE.md` 與明確任務引用載入。不假設工具會自動讀取任意四份 Markdown。
規範檔不等於作業系統沙盒；執行者另行設定檔案與網路權限，不能宣稱原始檔受 OS 強制保護。

Allowed：讀本案例、唯讀檢查、在新 run 計算、寫腳本與測試、提出假說。
Needs human decision：缺失的領域定義、改變目標／預測時點／切分、採納特徵假說、選定最終方案、業務處置。
Forbidden：改寫原檔、偷看 Holdout 選特徵、編造指標、複製憑證、把資料外傳、安裝系統套件、靜默換資料。
不建立額外的工具批准流程；明確授權範圍內的可逆工作繼續執行。

四階段有四個教學決策點，尚未批准的特徵可以討論但不進正式訓練。
審核記錄要包含學員實際決定，不能讓 AI 自己將 `proposed` 改成 `approved`。
規範、產物契約與四段提示詞詳見 [協作與交付契約](ai-workshop-contracts.md)。

## 9. 課程、預算與實作順序

完整時程、各單元作業及評分見 [教學與驗收](ai-workshop-teaching.md)。
預設 CPU、最多 4 threads、無 GPU；二手車教學樣本上限 12,000 列，依 proxy group 固定抽樣，
電力先用資料內連續最後 2 年並記錄起迄，乳癌使用全量。
預設每樹模型 200 estimators、不跑網格搜尋；計時超出課堂預算時減少 challenger 或樣本並另建 run，
不在 baseline／variant 間偷偷變更資料。8GB 記憶體作設計預算，實際耗時與峰值記憶體待實作量測。

實作批次順序：

1. 二手車垂直切片：資料契約、穩定列 ID、split manifest、EDA、Dummy／Ridge／LightGBM、指標。
2. 特徵登錄與消融、OOF SHAP、報告、四階段教學 runbook；建立可完整跑通的一題。
3. 乳癌 adapter、標籤方向、分類指標與局部錯誤案例。
4. AEP adapter、時間完整性、預測時點與順序回測；之後才加入 XGBoost challenger。

這是交付順序，不是已核准的程式實作計畫。架構審查後才細分改檔與測試步驟。

## 10. 驗收與回復

核心驗收是乾淨環境可重跑一題、三案例切分正確、原檔 hash 不變、結果可追溯。
實作必備測試：未來資料擾動不改變過去特徵；群組隔離；補值不讀驗證分佈；未知類別；
零分母／新車年份／缺檔；正類方向；同 Fold 配對；SHAP 加總與報告數值一致。
拒絕只測檔案存在、或複製實作公式的測試。詳見配套驗收清單。

既有課程完全沿原入口運作。新增 `workshop/` 與根 README 的一個入口即可導入，
不用移動資料或修改舊 notebook。後續只調整明確的 `workshop/workspace/` ignore；
本庫目前忽略 `*.json`、`figures/` 等，不能因此把新規範或契約藏進被忽略的目錄。
所有運算保留在新 run；若取消新工作坊，切回 main 即恢復既有路徑。
本次不自動 commit、push 或 merge。

## 11. 設計決策紀錄

- ADR-01：新增工作坊路徑，不原地重寫。替代方案與代價見第 4 節。
- ADR-02：一個離線可用主線＋兩個既有數值延伸；不新增陌生 Kaggle 題目。
- ADR-03：Markdown 宣告流程、Python 執行計算；不建立 Agent 調度服務。
- ADR-04：保留簡單模型作理解基準，Boosting 比較性能；SHAP 解釋預測，不推導因果。
- ADR-05：匯出矩陣不包含全表學得的轉換；Fold 內 fit 是計算邊界。
- ADR-06：不以強制提升或刪除特徵作及格條件；以證據品質與驗證誠實度驗收。

設計者可自行決定：腳本參數解析、配色、圖尺寸、函式拆分、序列化格式與錯誤訊息細節。
不能自行改動：目標定義、欄位可用時間、Holdout 邊界、SHAP 輸出單位及人類審查狀態。

## 12. 外部依據與查證限制

查證日 2026-10-28；官方 latest 文件用於方法依據，不代表本庫 lockfile 已包含其最新版 API。
實作時以鎖定版本做 smoke test，避免直接引用新 API。

- [scikit-learn：資料洩漏與 Pipeline](https://scikit-learn.org/stable/common_pitfalls.html)：先分資料，學得的轉換只 fit 訓練集。
- [scikit-learn：交叉驗證](https://scikit-learn.org/stable/modules/cross_validation.html)：依群組與時間結構選驗證方法。
- [LightGBM：Booster](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.Booster.html)：Gain 與 Split 定義不同，不能當成因果重要性。
- [SHAP：TreeExplainer](https://shap.readthedocs.io/en/latest/generated/shap.TreeExplainer.html)：輸出尺度、背景資料與特徵相依假設要明確。
- [SHAP：預測解釋與因果的區別](https://shap.readthedocs.io/en/latest/example_notebooks/overviews/Be%20careful%20when%20interpreting%20predictive%20models%20in%20search%20of%20causal%20insights.html)：透明的關聯不等於因果。
- [UCI WDBC 原始說明](https://archive.ics.uci.edu/dataset/17/breast+cancer+wisconsin+diagnostic)：569 筆、30 數值特徵，取自細胞核影像量測，原資料沒有缺失。

三個 Kaggle URL 可定位，但頁面動態內容未由瀏覽工具解析出 schema；欄位依本庫 CSV、
下載 manifest 與既有 notebook 交叉盤點。乳癌、AEP 目前未下載，下載後須再次驗證 schema／列數／hash，
不得把教材預期當成已讀取原始資料的證據。
