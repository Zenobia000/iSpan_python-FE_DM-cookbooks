# 案例、EDA、特徵與模型對照

> 案例方法參考附件。目錄與來源路徑反映整理時的舊專案狀態；新實作依 [Skill 架構](2026-10-28-skill-based-workshop-design.md) 的案例契約定位資料，不依賴舊例程仍存在。

本檔是[整體設計](2026-10-28-ai-feature-decision-workshop-design.md)的案例附件，狀態同主文件。
以下特徵全部是候選假說，尚未由學員核准、尚未計算效果。

## 1. 二手車：主要垂直切片

來源：[Kaggle UK Used Cars](https://www.kaggle.com/datasets/adityadesai13/used-car-dataset-ford-and-mercedes)。
目前可讀的路徑：`data_mining_course/projects/project/car_data/`。
只併 audi、bmw、ford、hyundi、merc、skoda、toyota、vauxhall、vw 九檔；品牌由檔名取得。
排除 focus、cclass 與 unclean 兩檔，不把重疊子集當獨立新資料。

欄位：`model, year, price, transmission, mileage, fuelType, tax, mpg, engineSize, brand`。
`hyundi.csv` 的 `tax(£)` 正規化為 `tax`，`model` 去首尾空白。
`price` 為英鎊、`mileage` 為英里、`engineSize` 為公升；`mpg` 的 gallon 制式仍須核對來源，未確認不換算成 L/100km。
`tax` 代表資料中的稅額，時間基準／制度需資料字典說明，不能跨年代直接等同持有成本。

決策：在已知刊登屬性時提供同類車刊登價格參考，供定價人員審查。
資料是歷史截面，沿用既有 capstone 的 `snapshot_year=2020` 作教學假設並明示，不能用執行當年算車齡。
若 `year > snapshot_year`，要求確認快照或隔離不合約列；不直接把負車齡剪成零。
price 不輸入模型、不建立 price/mileage 或由價格推導的等級特徵。

### EDA 要回答什麼

1. 品質：schema、空白 model、重複來源、非數值、缺失、零排氣量、負值與異常年份。
   電動車的零排氣量可能合理；異常只標記，不全部刪除。
2. 單變數：price／mileage 的 hist、ECDF、log 對照；各品牌筆數、長尾車型、類別缺失。
3. 雙變數：里程／年份與 price 的散點或 hexbin；品牌、燃料、變速箱的箱型圖，顯示樣本數。
4. 多變數：同品牌／同車型內車齡與里程關係，Pearson＋Spearman 熱圖；避免把車型組成差異說成單一因果。
5. 殘差：OOF 誤差按品牌、車齡、價格分位與稀有車型分組；價格分位只作誤差分析。

### 六個可審查特徵

| 名稱 | 公式／單位 | 假說與風險 |
|---|---|---|
| `vehicle_age` | 2020 - year，年 | 同快照下車齡反映折舊關聯，無法單靠資料證明折舊因果 |
| `mileage_per_year` | mileage / max(vehicle_age, 1)，英里／年 | 使用強度代理；新車統一分母下限 1，記錄近似 |
| `log_mileage` | log1p(mileage)，無因次轉換 | 壓縮長尾，對線性模型特別有價值；不接受負里程 |
| `age_mileage_interaction` | vehicle_age × log_mileage | 車齡與使用量聯合作用假說，不叫磨損實測量 |
| `engine_fuel_interaction` | engineSize × 各燃料指示 | 排氣量對不同燃料車型的價格關聯可能不同 |
| `model_frequency` | 訓練 Fold 內 brand+model 相對頻率 | 稀有性代理；驗證集未知車型設 0，不能讀全表頻率 |

前五項可作純列轉換，最後一項必須 fit Fold。接受其中少數或全部拒絕都可，需記錄理由。
樹模型本來可學交互，這些特徵可能沒有提升，必須做同模型配對比較。

### 模型與指標

必跑 DummyRegressor(median)、One-Hot＋補值＋縮放的 Ridge、LightGBMRegressor。
選跑 XGBoostRegressor、深度 3 決策樹。樹管線不強制標準化；Ridge 必須以訓練 Fold 學縮放。
主指標 MAE（英鎊），次指標 RMSE（英鎊）、R²；log1p(price) 僅作獨立實驗，
以 inverse transform 的預測回原尺度比較，揭露反轉換的估計偏差。
PCA 不列主線，因原始業務單位更利於解釋。

## 2. 乳癌：純數值分類與共線性

來源：[Kaggle WDBC](https://www.kaggle.com/datasets/uciml/breast-cancer-wisconsin-data)，
預期 `data_mining_course/datasets/raw/breast_cancer/data.csv`；實際下載檔名需核對。
排除 `id`，`diagnosis` 映射 M=1／B=0；若有 `Unnamed: 32` 必須先驗證是否全空，不能按名字盲刪。
預期為 569 列、30 個細胞核量測，包含 mean、se、worst 家族；不借用 sklearn 資料替代 Kaggle 卻沿用同一來源標籤。

EDA：類別比例；各特徵按診斷分組的箱型圖／ECDF；Pearson＋Spearman；
radius／perimeter／area 共線群；mean／worst 的關係；類別分組之樣本數與離群值。
原始 WDBC 無缺失；若要教缺失機制，另建立明示的擾動練習，不冒稱原始資料存在 MNAR。

候選特徵限 2–4 個，不強制 6 個：

- `radius_worst_to_mean`：worst／mean 尺度差異代理，不是時間變化率。
- `area_worst_to_mean`：同上，留意與 radius ratio 冗餘。
- `circularity_mean`：4π × area_mean / perimeter_mean²，無因次形狀代理，量測不一致時不保證落在 0–1。
- `texture_worst_minus_mean`：最極端與平均紋理差異，不解讀成病情進展速度。

零分母輸出缺失與指示，不加任意極小數製造巨大比值。
必跑 DummyClassifier(prior)、補值＋StandardScaler＋LogisticRegression、LightGBMClassifier；
選跑 XGBoostClassifier、淺樹；SVM 可比較但不強行用 TreeExplainer 解釋。
主指標 Average Precision（明確標為 AP，不混稱梯形 PR-AUC），次指標 ROC-AUC、
惡性 recall、precision、specificity、混淆矩陣、Brier score。
機率門檻預設 0.5 作共同對照；其他門檻須事前定義漏判成本或 sensitivity 約束，inner validation 選定。
人類要判斷漏判／誤報代價，而不是只看 Accuracy；不輸出可直接採用的臨床篩檢門檻。

## 3. 電力：歷史可見性與逐步回測

來源：[Kaggle Hourly Energy Consumption](https://www.kaggle.com/datasets/robikscube/hourly-energy-consumption)，
預期 `data_mining_course/datasets/raw/power_consumption/AEP_hourly.csv`。
欄位 `Datetime, AEP_MW`；不要與同目錄另一份 UCI household 電力混用，單位 MW／kW 不可混稱。

定義一列的索引為預測起點 t，發報時已觀測 `AEP_MW(t)`，目標 `AEP_MW(t+1)`。
已知曆法可用 t+1 的 hour／weekday；歷史負載只能到 t。
若改用標籤時間 s=t+1 作索引，全部負載特徵必須先 shift(1)，兩種索引慣例不可混搭。

EDA：排序、重複時間、缺時與時差分佈、時區／DST 歧義；負載時間圖、每日／每週箱型圖、
hour×weekday 熱圖、ACF、lag plot、跨季節分佈。只在開發集分析，
雙向插值與 centered rolling 不用於預測；季節分解只作描述，不把使用未來資訊的趨勢項餵模型。
未知時區／重複 DST 小時要停下確認，不盲目平均成一小時。

候選：目前負載、前 23／167 小時負載（對應目標時刻的昨日／上週同期）、
截至 t 的 24h／168h 平均、24h 標準差、當前相對前 24h 的差、目標時刻 hour 的 sin／cos。
每個都寫清輸入時間範圍；rolling 只使用完整過去窗口。
歷史不足的列統一剔除或依合約處理，baseline／variant 使用相同的可評估列。

固定完整小時網格後做位置 lag；缺時不能把「前一列」冒稱「前一小時」。
預設只對目標與最長所需歷史窗口完整的列評估，紀錄剔除比例與樣本選擇限制。
為保證訓練標籤不超過第一個驗證起點的前一刻，一小時 horizon 預設 purge 末 1 個訓練起點。
任何 horizon／觀測延遲改動都以時間戳重新計算 purge，不盲目把窗口長度當 gap。

基準：預測目標時刻使用 AEP(t-23)、AEP(t-167)，另列 persistence AEP(t)。
必跑季節基準、Ridge、LightGBMRegressor；XGBoost 選修。
主 MAE（MW）、次 RMSE（MW）、相對最強季節基準改善；不以 MAPE 當唯一指標。
驗證採 rolling one-step：每小時新實際負載公布後可供下一次預測，模型保持該 Fold 訓練版本。
不能在一次 24h batch forecast 中使用尚未發布的測試真值；多步預測另立實驗。

## 4. EDA 共通規格與處置界線

| 問題 | 方法 | 可以決定什麼／不能決定什麼 |
|---|---|---|
| 缺失與型態 | 缺失率、missingness pattern、字典對照 | 提出缺失機制假說；不能從缺失率證明 MCAR／MAR／MNAR |
| 偏態與尾部 | skew、excess kurtosis、分位數、hist／ECDF | abs(skew)>1 標記檢視；不自動 log 所有欄位 |
| 極值 | IQR、MAD、領域範圍、分群箱型圖 | 區分測量錯與稀有事件；不因模型不喜歡就刪除 |
| 相依性 | Pearson、Spearman、散點 | abs(r)>0.85 標記；不自動刪其中一個，更不推導因果 |
| 類別與群組 | 類別頻次、未知類別、分群樣本數 | 查稀有性與偏差，保留低樣本警示 |
| 時間 | 缺時、ACF、季節箱型圖、漂移 | 檢查可預測性與發報條件，不能憑週期圖宣稱外部原因 |

恆定欄位的相關與 skew 可能未定義，記為未定義，不補成 0。
數據不足以判定缺失機制時明寫未知；需領域、收集流程與額外證據。
溫度攝氏、變化量、SHAP 均可能為負；物理範圍必須逐欄指定，禁止一律非負。

## 5. 舊教材入口

- [課程宗旨](../../../data_mining_course/docs/course_charter.md)
- [系統化 EDA](../../../data_mining_course/modules/module_01_eda_intro/slides/EDA_Systematic_Workflow_Guide.md)
- [二手車總整](../../../data_mining_course/capstone/capstone_used_car.ipynb)
- [乳癌特徵選擇](../../../data_mining_course/modules/module_07_feature_selection/notebooks/05_breast_cancer_case.ipynb)
- [電力完整案例](../../../data_mining_course/modules/module_08_time_series/notebooks/05_power_consumption_case.ipynb)
- [Telco 樹模型](../../../data_mining_course/modules/module_10_data_mining_applications/notebooks/03_tree_models/03_telco_churn_case.ipynb)
