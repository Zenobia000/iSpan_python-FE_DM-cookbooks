# 以功能與 Skill 套件重建的工作坊架構

> 歷史提案：其「五個獨立 skill」方案已由單入口 [scientific-analysis](../../../.agents/skills/scientific-analysis/SKILL.md) 取代。此文件保留決策脈絡，不作現行命令或能力清單。實際支援及未完成範圍依 skill、CLI 與 [執行紀錄](../../mvp-execution.md) 為準。

日期：2026-10-28。狀態：設計提案，尚未建立新 skill 或腳本。
本文件取代前版「集中 workshop/scripts＋四個階段規範」的封裝方案。
前版驗證、案例、課時與人機審查原則可繼續參考；路徑、元件責任與視覺化選型以本文件為準。
舊技術範圍已獨立保存於 [舊版課程知識大綱](../../legacy-curriculum-outline.md)，不要求保留舊程式目錄。

## 1. 核心判斷

以「能完成什麼工作」定義 skill，將規範、專屬程式、參考資料與模板一起封裝。
課程單元是組合這些能力的教學順序；不能照舊章節一章複製成一個 skill。
第一版建立五個功能 skill；必要的共用 Python 核心僅有一份，不複製切分與指標邏輯。

AI 負責把自然語言任務轉成有型態、可驗證的工作規格，腳本負責確定性計算，
人類審查問題、假說、可用資訊、模型選擇與行動。不是新增 Agent 調度系統。

## 2. Archify 的實際參考與取捨

已讀取本機另一專案的 `.claude/skills/archify/SKILL.md`、目錄與 CLI；
該版本 frontmatter 為 2.17，package 為 `2.17.0-dev.1`，不當成上游當前穩定版。
本機原件：`/home/sunny/python_workstation/github/AI_Content_Operating_System/.claude/skills/archify/`。
其他環境可參考 [上游 Archify](https://github.com/tt-a1i/archify)，本專案不依賴該絕對路徑。

| Archify 的設計 | 本專案採用方式 |
|---|---|
| `SKILL.md` 是入口與選擇規則 | 描述適用任務、輸入、步驟、限制與交付，不把全部技術文件塞進入口 |
| 型別化 JSON 規格＋schema | 用版本化 TaskSpec／ArtifactManifest 連接 AI 與腳本，拒絕不存在的欄位與矛盾目標 |
| CLI＋專屬 renderer | 每個 skill 自帶可執行入口，數值運算與繪圖交給 Python 套件 |
| references／examples／assets 按需載入 | 方法選擇放 references，schema 範例少量保存，報告模板放 assets |
| validate／deliver 分開 | 先檢查輸入與輸出，再以新 run 交付，不覆蓋已驗收的產物 |
| receipt＋視覺檢查 | 保存版本、hash、列數與產物索引；數值正確和圖表可讀分開驗證 |

不照搬 renderer 引擎、路由解算、動畫、WebM、品牌圖示、更新檢查或 Node 工具鏈。
本課需要的是資料科學功能封裝；只有使用者另要架構展示圖時才考慮呼叫 Archify。
本輪是研究其封裝模式，沒有呼叫 Archify 產圖，也不執行它的更新器。

## 3. 功能清單與 Skill 分工

| 能力／組件 | 功能 | 交付 | 舊知識對應 |
|---|---|---|---|
| `tabular-diagnosis`：schema／品質 | 來源、型態、單位、缺失、重複、時間／群組檢查 | data_profile、品質問題 | M01–M03 |
| 同 skill：EDA／假說證據 | 單／雙／多變數、分群、相依性、時序特性 | EDA 圖與診斷摘要 | M01、M03、M08 |
| `feature-engineering`：純轉換 | 欄位正規化、ratio、delta、日期、交互 | 版本化特徵與字典 | M02、M06 |
| 同 skill：學得轉換 | 補值、編碼、縮放、群組統計、時序視窗 | 可 fit／transform 的元件與邊界測試 | M03–M06、M08 |
| `model-evaluation`：驗證 | 固定 Fold、Pipeline、分類／回歸／時序指標 | split manifest、OOF、逐 Fold 結果 | M07、M08、M10 |
| 同 skill：比較／定案 | baseline、challenger、配對差、消融、finalize | 比較結果、最終模型與一次 Holdout | M07、M10、總整 |
| `model-explanation`：模型歸因 | 係數／規則、Gain／Split、permutation、OOF SHAP | 歸因數據、尺度與穩定性 | M07、M10＋新增 XAI |
| 同 skill：局部／群組錯誤 | 個案分解、誤差群組、候選轉折區間 | 誤差與局部解釋圖 | 新增決策診斷 |
| `decision-report`：證據組裝 | 比較、假說採否、歸因、限制與行動 | Markdown＋可選離線 HTML | 總整＋新增交付契約 |

組件是 skill 內部函式／模組，不因此各建一個 skill 或服務。
例如 one-hot、median imputer、boxplot 都只是組件，不是三個獨立技能。
第一版不新增通用「全自動分析」skill；課程 runbook 或使用者直接組合五個能力。

### 責任邊界

- diagnosis 不覆寫原始資料或全表補值，依已定義的 split contract 只探索開發集。
- feature-engineering 專管特徵公式與 transformer，不自行選最佳模型或判定因果。
- model-evaluation 專管 fit、指標、permutation 與消融；特徵篩選在 inner split，不能偷看 Holdout。
- model-explanation 讀已 fit 模型與 OOF 資料；可請求 evaluation 產生額外診斷，但不自行重訓。
- decision-report 不重新計算指標或訓練模型，只組合已驗證證據與人類決定。

診斷的分群／EDA 與報告的圖表不同：前者產生分析證據，後者只排列或導覽已有證據。
統計圖由產生該證據的 skill 擁有；共用風格元件放核心，不建立第六套重複計算的 visualization skill。

## 4. 目標封裝

下列是待實作目錄。技能以 `.agents/skills/` 為唯一原始碼來源；其他工具的發現機制需實機驗證。

```text
.agents/skills/
  tabular-diagnosis/
    SKILL.md
    scripts/run.py
    references/eda-methods.md
  feature-engineering/
    SKILL.md
    scripts/run.py
    transforms.py
    references/feature-rules.md
  model-evaluation/
    SKILL.md
    scripts/run.py
    estimators.py
    references/validation.md
  model-explanation/
    SKILL.md
    scripts/run.py
    explainers.py
    references/explanation-limits.md
  decision-report/
    SKILL.md
    scripts/run.py
    assets/report.md.j2
    assets/report.html.j2
src/fe_workshop/
  contracts.py               # TaskSpec、ArtifactManifest 與驗證
  data.py                    # 唯讀來源、row identity 與資料契約
  splits.py                  # 固定分割，供診斷與模型評估使用
  artifacts.py               # run／stage 產物、hash 與不覆寫交付
  plotting.py                # 圖表風格、標籤與匯出薄封裝
cases/<case-id>/
  data_contract.md
  feature_registry.md
  experiment_plan.md
  decisions.md
docs/
  legacy-curriculum-outline.md
  workshop-runbook.md
tests/                       # 共用契約與跨 skill 整合測試
workspace/<case-id>/<run-id>/ # 資料衍生物、模型、圖與報告，不進版控
AGENTS.md
CLAUDE.md
```

只在確實需要時增加 `schemas/`、`examples/`、`tests/` 等 skill 內資源，不建空目錄。
SKILL.md 的 name／description 精確描述觸發情境；body 提供必備決策與相關 references 連結。
不把完整資料、模型權重、一次實驗報告放進 skill，避免使用次數越多套件越肥。

### 共用核心與可攜性

五個 skill 是本專案內可獨立呼叫的能力套件，依賴同一份 `fe_workshop` 核心與專案鎖定環境。
這不是「任意複製單一資料夾就完全獨立」的承諾；若日後需跨 repo 安裝，再設計核心套件發佈。
只在有兩個以上實際呼叫者時抽出共用程式，不為未來想像建立 plugin registry。
skill 間不得 import 對方 scripts；透過核心契約與產物交接，AI 讀文字也能理解如何續接。

保留現有 uv 工具鏈。實作時將根專案設成可安裝的本地 `fe_workshop` 套件，
採明確 build backend 與 src discovery；保留既有依賴直至清理範圍確定。
skill CLI 在專案虛擬環境 import 核心，不散落 `sys.path` 修補，也不靠任意工作目錄猜相對路徑。
這是對現有 `package=false` 的預定變更，目前沒有修改 pyproject／lockfile；需在實作時驗證。

### 工具入口

AGENTS.md／CLAUDE.md 提供相同的能力目錄與資料邊界，不複製 skill 正文。
Codex 使用專案技能發現；若 Claude 環境要求 `.claude/skills/`，以薄入口或安裝時連結接入，
實機驗證描述、相對資源路徑與觸發後再交付。不以兩份可編輯正文維持同步。
即使沒有自動發現，也可明確要求 AI 閱讀指定 SKILL.md，再執行其腳本。

## 5. 規格、產物與流程

人類提供目標與約束，AI 提出 TaskSpec，腳本先驗證再運算，交付 ArtifactManifest，
人類在決策點採納或否決。TaskSpec 是工作參數與允許的轉換名稱，不接受可執行 Python 字串。
新領域公式需寫成可審查的 Python 轉換並驗證，不建任意 `eval` 運算介面。

共用 TaskSpec 最少欄位：schema_version、case_id、run_id、stage、資料契約版本／hash、
target／task type、prediction_time／horizon、split policy／manifest 引用、feature_set、
主指標、seed、budget、前階段產物引用及人類決定引用。
每個 skill 加自己的參數驗證；未知鍵、缺目標、未批准特徵或任務類型矛盾需清楚報錯。
schema 不只驗 JSON 型別，也要驗資料欄位存在、時間可見性與配對實驗列集合。

ArtifactManifest 最少包含輸入 hash、skill／程式版本、TaskSpec hash、環境、實際參數、
資料列／Fold 涵蓋、耗時、產物路徑／型別／hash、警示與驗證狀態。
必要產物：diagnosis profile；features baseline／v1；split manifest；metrics_cv＋OOF；
explanation values／metadata；final metrics＋report。學得轉換不寫入全表 feature CSV。
每階段新寫自己的 manifest，不改寫前階段輸出；已存在就拒絕覆寫。

四個人類決策點保留：G1 資料含義、G2 特徵假說、G3 最終方案、G4 業務結論。
五個 skill 不等於五次固定批准流程；可在原授權內持續完成計算，只有真實教學決策才停下。
G3 後 evaluation finalize 全開發集重訓並驗收一次 Holdout；report 不訓練或重測。
簡單模型勝出時改用係數／規則解釋；不能為了 SHAP 圖偷換最終模型。

資料案例獨立於 skill：二手車、乳癌、AEP 各自描述 schema 與領域假說，同一能力重用。
目前二手車資料位於舊專案；移除舊目錄前必須先確定保留或重新取得的資料方案。
新實作以案例資料契約定位來源，不把舊絕對路徑寫死於技能；缺資料就明確停下。

## 6. 視覺化：優先套件，按任務補少量自訂

| 任務 | 選型 | 決策 |
|---|---|---|
| 分佈、類別比較、相關性、時間圖 | Seaborn＋Matplotlib | 核心；沿用既有依賴與成熟統計繪圖 |
| 缺失模式 | missingno；簡單缺失率可用 Seaborn | 已有依賴，按需使用，不宣稱能判定缺失機制 |
| ROC、PR、混淆矩陣、校準、預測誤差 | sklearn Display API | 核心，按任務挑圖，不重寫指標計算 |
| 樹模型全局／局部歸因 | SHAP 原生 plots | 以鎖定版驗證；保留 Explanation／尺度 metadata |
| hover、zoom、分群切換、離線交互 | Plotly | 選用 viz extra，HTML 內嵌 plotly.js，不預設 CDN |
| 快速全表 profiling | ydata-profiling | 選修候選；先驗相依與資源成本，僅開發集，不能取代領域 EDA |
| 季節／自相關診斷 | statsmodels＋Matplotlib | 時序模式按需使用，不另寫分解算法 |
| Fold 配對差、消融前後、證據索引 | 小型自訂資料整理＋現成繪圖 | 本課有特定語意，值得自訂，但不造新繪圖引擎 |

離線 HTML 報告使用薄模板，把現有圖、表與證據引用組起來；可選 Jinja2 作模板依賴。
Markdown 是核心交付，HTML 為選用；不為展示建立 Dash／Streamlit 常駐服務。
Plotly 靜態圖片匯出可能增加 Kaleido／瀏覽器需求；第一版 PNG 仍由 Matplotlib／SHAP 產生，
Plotly 只負責互動 HTML，免為匯出再增加一條執行鏈。
所有新增依賴先確認與現有 NumPy／Python 範圍相容，再更新 uv.lock；本次不安裝。

### 必須自訂的薄層

統一字體／單位／色彩、圖名、樣本數、資料範圍、抽樣方式與 caption；
配對比較的差值方向、跨 Fold 誤差條與消融標記；圖表到實驗與人類決定的連結。
統計數值先保存成結構化產物，再繪圖，報告不從圖片猜數字。
對數軸與 SHAP raw output 的單位須明示；不存在的門檻不為視覺效果硬畫。
超大資料先按固定 seed 抽樣或聚合並標註；不把幾百萬筆原始資料塞進 HTML。

### 驗證分兩種

數值：圖表資料與 metrics／OOF 對齊、群組數量正確、尺度與正類方向一致。
視覺：圖例／標籤不裁切、中文可讀、單位／樣本數完整、HTML 離線可用。
有圖片檔案不等於已通過這兩種檢查。

## 7. 重建順序與驗收

1. 先保存舊知識大綱，供刪除舊例程後參考；不先複製全部舊程式進新技能。
2. 實作最小共用契約＋diagnosis，驗證一份二手車資料與幾張有決策意義的圖。
3. 串 feature-engineering＋evaluation：Dummy／Ridge／LightGBM 同 Fold、原始／新特徵配對。
4. 補 explanation＋report，完成一題可重跑的垂直流程；再加入乳癌與電力 adapter。
5. 需要時才補 Plotly HTML／profiling、XGBoost、target encoding 等進階組件。

每個 skill 的 description 可正確觸發；SKILL.md 可找到實際腳本與參考；
腳本在支援的本地環境可直接執行，輸入不合約時報錯，不靜默生成合成 Kaggle 資料。
整合驗收檢查 Fold 內 fit、群組／時間隔離、未來擾動不影響過去、SHAP 加總、
配對指標方向、報告數值對應與原檔 hash 不變。
移除舊例程後仍可按新入口完成教學；這是相容性驗收的目標，不是保留舊 notebook 的要求。

本次只交付知識大綱與重設計文件；沒有刪檔、建立技能實作、移動資料或安裝套件。

## 8. 查證來源

查證日 2026-10-28。方法與能力可由官方文件確認，但未做新增套件的安裝／相容性測試。

- [Archify 上游](https://github.com/tt-a1i/archify)：結構化規格、渲染與交付驗證的參考。
- [Seaborn 官方教學](https://seaborn.pydata.org/tutorial.html)：分佈、類別與統計關係圖。
- [sklearn 視覺化 API](https://scikit-learn.org/stable/visualizations.html)：Display 與模型診斷。
- [SHAP API](https://shap.readthedocs.io/en/latest/api.html)：Explanation、explainer 與原生 plots。
- [Plotly HTML 匯出](https://plotly.com/python/interactive-html-export/)：瀏覽器互動與自包含匯出。
- [YData Profiling 官方文件](https://docs.profiling.ydata.ai/latest/)：profiling 的候選能力與設定依據。
