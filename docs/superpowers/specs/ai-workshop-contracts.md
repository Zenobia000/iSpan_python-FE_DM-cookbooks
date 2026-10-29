# 四階段協作規範與交付契約

> 前版規範草案。四個人類決策點與驗證原則保留；集中式 `workshop/scripts`、`.claude/01_eda.md` 等封裝及命令介面已由 [新版 Skill 架構](2026-10-28-skill-based-workshop-design.md) 取代，不照本檔舊路徑建置。

本檔是[整體設計](2026-10-28-ai-feature-decision-workshop-design.md)的規範設計稿。
以下 `workshop/` 路徑與指令為待實作介面，現在不可直接執行。
不以 Markdown 文字聲稱 OS 已限制原始資料的寫入權限。

## 1. 前置契約

每個案例先備妥 `data_contract.md`：來源 URL、檔案路徑與 hash、schema、單位、
target 與正類映射、預測時點／horizon、ID／時間／群組、允許／禁止特徵、
split policy、holdout policy、領域範圍、缺失含義、資料限制與版本。
來源不明的欄位、缺檔、目標不存在或正類不清楚時停止依賴該資訊的計算。

`feature_registry.md` 每列至少含：

| 欄位 | 內容 |
|---|---|
| feature_id／name／version | 穩定識別與版本 |
| parents／formula／unit | 父欄位、公式、量綱 |
| hypothesis／expected_direction | 領域理由、預期方向或未知 |
| available_at／fit_scope | 資訊可用時間；row_pure／train_fold |
| missing_policy／zero_policy | 缺失、分母為零與邊界處理 |
| status／reviewer／decision_ref | proposed／approved／rejected；真實審查者與理由 |
| evidence_ref | 事後連到消融、Fold 指標與圖表 |

`experiment_plan.md` 在訓練前凍結：主次指標、方向、模型與參數、樣本預算、
特徵組、split hash、seed、threads、選擇規則與停止規則。
`decisions.md` 保留學員原始判斷、否決原因與日期，AI 不能自行簽核。

## 2. 01_eda.md：數據診斷規範設計

角色：資料診斷與假說整理助理。

輸入：已審查的案例資料契約、唯讀原檔、已建立的開發集／Holdout manifest。

任務：

1. 執行 schema、單位、非有限數、ID、重複列、時間粒度檢查，記錄資料版本。
2. 在開發集做品質、單變數、雙變數、多變數探索；預定義規則的全體品質檢查不涉及標籤探索。
3. 計算缺失率、分位數、skew、excess kurtosis，Pearson／Spearman 相依性。
4. 依案例選圖，不生成全部欄位 pairplot 或無目的的大圖。
5. 列前三大資料問題、證據、可能含義與待人類決定事項。沒有三個問題不硬湊。

邊界：不改原檔、不補全表、不自動刪離群值、不斷言 MCAR／MNAR、不讀 Holdout 目標做決策。
缺失機制要標明「觀察／假說／未知」。圖中抽樣、對數軸、單位與樣本數必須註明。

產出 `output/eda_report.md`、`eda_profile.json`、`figures/eda_correlation.png`、
`eda_target.png`，加案例需要的分佈／群組／時序圖。
教學決策點 G1：學員確認資料含義、異常處置與切分是否符合用途。

## 3. 02_feature_eng.md：領域特徵規範設計

角色：領域假說與特徵實作助理。

輸入：G1 決定、資料契約、特徵登錄表與開發集 EDA。

任務：

1. 提出少量可檢驗的特徵，二手車約 6 個、乳癌 2–4 個、時序依預測時點決定。
2. 寫出公式、量綱、可見資訊、缺失／零分母政策與預期可推翻的結果。
3. 在學員核准後實作純列轉換與 Fold 內 transformer；兩者不能混在全表輸出中。
4. 時序按實體分組、排序與正確時間網格計算，只使用發報時已知的資料。
5. 將邏輯固化於 `scripts/build_features.py` 與 `lib/features.py`，加實際邊界測試。

邊界：禁止 bfill、雙向插值、未來視窗、以 target 造特徵、全表群組統計與無理由多項式。
缺失插值即使合理也不能跨資料邊界使用未來值；預設 Fold 內中位數／類別補值＋指示，
或嚴格因果的 forward fill（同實體、最大容許時間間隔已定義）。不以分佈名稱自動決定補值。
不一律要求非負；每個變數的物理範圍由資料契約定義。

輸出 `processed/features_baseline.csv` 與 `features_v1.csv`，保留 row_id、target、time／group metadata。
這兩檔只含原始欄位與不需 fit 的安全衍生；metadata／target 不進模型 X。
缺失尚可存在；model_frequency 等學得的特徵在 train_eval 的 Fold 內生成，並列入 feature dictionary。
匯出矩陣不冒稱最終已補值、已編碼的部署矩陣。
教學決策點 G2：學員核准／否決假說，更新登錄表；未核准特徵不得進正式比較。

## 4. 03_modeling.md：驗證與消融規範設計

角色：模型驗證助理。

輸入：核准的特徵組、固定 split manifest、experiment plan；不以 CSV 全部欄位當 X。

任務：

1. 載入既定 Fold，檢查列集合、時間／群組隔離與樣本數。
2. 執行 Dummy／季節基準、簡單模型、LightGBM，以及選修 XGBoost。
3. 每 Fold 重建並 fit Pipeline；原始／新特徵採同模型參數、同 Fold 與同列集合。
4. 輸出 OOF 預測、逐 Fold 指標與配對差，計時並記錄模型與環境。
5. 比較 Gain／Split、驗證集 permutation 與特徵群消融；特徵篩選放 inner split。
6. 依預先選擇規則提出候選，允許回退到簡單模型或 baseline。

邊界：不在外層驗證集 early stop、不全表 fit、不用 Accuracy 掩蓋漏判、
不只回報最好 Fold、不保證提升、不用同一 Holdout 反覆挑選。
Permutation 在相關特徵下可能低估價值，需以特徵群與消融對照。

CV 階段產出 `output/metrics_cv.json`、`oof_predictions.csv`、`feature_importance.csv`、
`ablation_results.csv`、`figures/feature_importance.png`、`model_comparison.png`，
以及 `artifacts/<model>/<variant>/fold_<k>/` 中的完整 Pipeline 與字典。
教學決策點 G3：學員依指標、穩定性、成本與可解釋性凍結最終模型／特徵／門檻。
G3 後另執行同腳本的 `--stage finalize`：讀取凍結決定、全開發集重訓所選 Pipeline、
儲存 `artifacts/final_pipeline`，一次評估 Holdout，建立 `output/metrics.json`。
該檔整合 CV 指標、selection 與 Holdout，引用原始 `metrics_cv.json` 的 hash；不改寫 CV 結果。

## 5. 04_interpretability.md：歸因與報告規範設計

角色：XAI 與證據整理助理。

輸入：每 Fold 的已訓練模型／前處理器、OOF 列與抽樣規則、metrics、G1–G3 記錄。

任務：

1. 依主設計第 7 節計算並驗證 Tree SHAP，簡單模型另用係數／規則解釋。
2. 在 OOF 資料作 Summary、前兩名 Dependence、典型與錯誤案例的局部圖。
3. 按原始業務欄位整理 One-Hot 歸因，保留特徵間相依與稀有樣本限制。
4. 用跨 Fold 分箱與樣本數查候選轉折，沒有穩定轉折就不提單點門檻。
5. 區分事實、模型歸因、領域假說與待驗證行動，將每個數值連到產物。

最終為樹模型時產出 `figures/shap_summary.png`、`shap_dependence_top1.png`、`shap_dependence_top2.png`、
兩張 local explanation、`shap_metadata.json` 與 `output/final_decision_report.md`。
最終為 Ridge／Logistic／淺樹時，必要圖改為係數／規則與局部解釋；LightGBM SHAP 僅作
具名比較模型的選修產物。不用 TreeExplainer 解釋線性模型，也不為湊圖偷偷替換已選模型。
G3 凍結後由 train_eval finalize 產生的一次 Holdout 指標另列，不混入 OOF 歸因或候選門檻選擇。
報告層不訓練、不評估 Holdout；OOF 歸因標示為 Fold 模型，與全開發集重訓模型區別。
教學決策點 G4：學員簽核報告中的適用範圍、假說與後續驗證行動。

## 6. Run manifest 與 metrics 契約

每次 run 的 `output/run_manifest.json` 必須包含：

- schema_version、run_id、case_id、UTC 時間、Git SHA、dirty 狀態／diff hash。
- 原檔相對路徑、SHA256、schema、列數、抽樣／去重政策；不得包含憑證。
- 特徵登錄與契約 hash、split manifest hash、各 Fold row_id／group／time 範圍。
- 目標／正類、預測時點、horizon、purge、內外層切分、holdout 存取狀態。
- seed、預定參數、threads、Python／套件版本與預期產物路徑。

初始 run_manifest 建立後唯讀；各階段另外建立 `output/stages/<stage>_manifest.json`，
記錄輸入 hash、實際模型參數、耗時、產物清單與 hash，避免回寫前階段 manifest。
stage 使用 validate／eda／features／cv／finalize／report，存在時拒絕重新執行該階段。

`output/metrics_cv.json` 與最終 `metrics.json` 使用 `schema_version: 1`，主要欄位：

| 區塊 | 必備內容 |
|---|---|
| task | target、正類／單位、主指標、greater_is_better |
| experiments[] | model、feature_set、params、fit_count、runtime |
| folds[] | fold_id、n_train／n_valid、時間範圍、各指標、OOF 檔案引用 |
| aggregate | mean、std、每 Fold 結果；時序另列各時段 |
| paired_comparisons[] | 相同 split hash、model、每 Fold raw_delta 與 improvement、樣本數 |
| selection | 凍結方案、選擇理由、decisions 引用 |
| holdout | 未存取／一次驗收狀態、模型、凍結時間與測試指標 |

CV 檔的 selection／holdout 僅記 pending／unaccessed；finalize 新建最終 metrics，不改寫原檔。
finalize 在存取 Holdout 前先記錄不可變的評估開始標記；若執行失敗，不容許換方案再試，
僅可依原凍結方案人工審查後恢復，且記錄恢復事實。

非有限數值不寫 JSON NaN／Infinity，使用 null 與原因。
OOF CSV 至少含 row_id、fold_id、model、feature_set、y_true、y_pred、分類時 y_score、
時序時 origin_time／target_time。時間序列 warmup 未涵蓋列不得偽裝成 OOF 預測。
圖與報告的數值只讀上述產物；LLM 可潤稿，不能重新估算或補不存在的指標。
OOF 中含標籤，只供評估／報告，不得回流特徵生成。

## 7. 最終報告模板

```markdown
# <案例> 特徵工程與模型決策報告

## 決策問題與適用範圍
目標、資訊截止時間、業務用途、資料限制。

## 資料與驗證證據
run_id、來源與 hash、樣本、Holdout、Fold、洩漏檢查、排除規則。

## 實驗比較
簡單基準、原始特徵模型、新特徵模型，逐 Fold 與整體分數；
原始差值與改善方向、耗時、消融結果。無改善如實寫。

## 特徵假說的採納與否決
公式、單位、可用時間、審核者、支持或推翻假說的證據。

## 模型歸因
Summary、前二名 Dependence、局部錯誤與典型案例；
輸出尺度、抽樣涵蓋、相關特徵與排名穩定性。

## 候選區間與待驗證行動
有證據才列候選區間，附樣本與跨 Fold 對照；
分類門檻獨立說明成本、選擇資料與誤報／漏判。
分開列觀察、模型關聯、領域假說、待驗證行動。

## 一次 Holdout 驗收與限制
凍結方案、最終分數、群組差異、不適用場景及後續驗證。

## 重現方式與產物索引
環境版本、完整命令、manifest／metrics／圖表相對連結。
```

## 8. 互動式 Runbook 設計

開始前讀 `workshop/AGENTS.md` 或 `CLAUDE.md` 與對應案例契約。
人類用語音或自然語言下指令即可，不必先學 workflow DSL。

1. 「讀取 `.claude/01_eda.md` 與 used_cars/data_contract。先驗證來源與切分，再在開發集
   做 EDA；提出三個最需審查的品質問題。不要修改原檔。」學員完成 G1。
2. 「讀取 `.claude/02_feature_eng.md`，依 G1 提出六個特徵及可推翻的假說；
   先讓我審核，再實作核准項目與邊界測試。」學員完成 G2 後才計算正式特徵。
3. 「讀取 `.claude/03_modeling.md`，固定同 Fold，比較 Dummy／Ridge／LightGBM，
   再以同參數比較 baseline 與 v1，輸出逐 Fold 指標、OOF 與汰除候選。」學員完成 G3。
4. 「依 G3 凍結決定執行 train_eval finalize，重訓並一次驗收 Holdout。
   再讀取 `.claude/04_interpretability.md`，以 OOF 計算所選模型適用的歸因，
   標清尺度與限制，讀取最終指標產出報告。」學員完成 G4。

預定實作後命令介面（從 repo root 執行，現在尚不存在）：

```bash
uv run python workshop/scripts/validate_data.py --case used_cars --run-id <run_id>
uv run python workshop/scripts/run_eda.py --case used_cars --run-id <run_id>
uv run python workshop/scripts/build_features.py --case used_cars --run-id <run_id>
uv run python workshop/scripts/train_eval.py --case used_cars --run-id <run_id>
# Run only after the human freezes the selection at G3.
uv run python workshop/scripts/train_eval.py --case used_cars --run-id <run_id> --stage finalize
uv run python workshop/scripts/explain_report.py --case used_cars --run-id <run_id>
```

validate_data 建立新 run 與初始 split，其餘命令僅接續同一 run 的預定階段；
目標階段產物已存在時拒絕覆寫。更改契約／特徵／預算就建立新 run 並引用 parent_run_id。
既有下載器會嘗試自行安裝套件；新工作坊不可直接依賴這種行為，下載僅限已備好的專案環境，
缺依賴即提供 uv 設定指引。規則不應放憑證內容、病人資料或私人資料樣本。
