# 模組3 缺值與異常值：投影片規劃

9 張，走 `handout_blueprint.md` 的四格骨架（道 1 · 術 3 · 法 3 · 器 2）。
講義本體 [`M3_Missing_and_Outliers_Fundamentals.md`](../../modules/module_03_missing_outliers/slides/M3_Missing_and_Outliers_Fundamentals.md)，產圖提示詞在 [`slides.prompts.md`](slides.prompts.md)，圖檔輸出到 `concept_images/`。

落在六步的第 5 到 6 步。這堂對應的鐵律是鐵律一（先切再轉）與鐵律二（講人話）。

| # | 格 | 標題 | 這張要呈現什麼 | 對應 notebook | 圖 |
|---|---|---|---|---|---|
| 1 | 道 | 這堂在解什麼 | 缺值本身可能就是訊號。先講「為什麼缺」再講「怎麼補」，AI 可以寫插補程式，人要判斷缺失機制 | `01_missing_data_overview.ipynb` | 圖 1 |
| 2 | 術 | 第一原理 | 缺失機制決定能不能補。MCAR、MAR、MNAR 三種各自代表什麼，MNAR 補了等於捏造 | `01_missing_data_overview.ipynb` | |
| 3 | 術 | 決策表 | 刪除／均值／中位數／KNN／加指示欄五種方法，比適用情境、代價、洩漏風險、`fit` 學到什麼 | `02_imputation_methods.ipynb` | 圖 2 |
| 4 | 術 | 判斷表 | 依欄位型別與缺失比例決定策略；缺很多的欄位什麼時候該整欄砍掉 | `02_imputation_methods.ipynb` | |
| 5 | 法 | SOP 五步 | 看缺失比例 → 判斷機制 → 選方法 → 只用訓練集 `fit` → 加指示欄保留訊號 | `04_house_prices_case.ipynb` | |
| 6 | 法 | 驗收三問 | 人話、洩漏、拿掉會怎樣。異常值多一問：這是錯誤還是真實的極端值，兩者處理方式相反 | — | |
| 7 | 法 | 反例 | 用全資料算中位數再切分，測試分數虛高。附錯誤程式與前後分數 | `03_outlier_detection.ipynb` | 圖 3 |
| 8 | 器 | API 速查 | `SimpleImputer`、`KNNImputer`、`IterativeImputer` 的最小可跑範例，含 `add_indicator` | `02_imputation_methods.ipynb` | |
| 9 | 器 | 三個陷阱與 notebook 對照 | 陷阱：對測試集重新 `fit`、把 MNAR 當 MCAR 補、用平均值補嚴重偏態欄位 | 全部 | |

要學員記住的那句話：缺值不是要填滿的空格，是要先解釋的現象。

對應 notebook 都在 `../../modules/module_03_missing_outliers/notebooks/` 底下。
