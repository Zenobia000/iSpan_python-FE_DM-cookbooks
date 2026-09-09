# 第 5 章　特徵縮放與轉換：投影片規劃

9 張，採 `handout_blueprint.md` 的四格骨架（道 1 · 術 3 · 法 3 · 器 2）。
講義本體 [`M5_Scaling_and_Transformation_Fundamentals.md`](../../modules/module_05_scaling_transformation/slides/M5_Scaling_and_Transformation_Fundamentals.md)，產圖提示詞在 [`slides.prompts.md`](slides.prompts.md)，圖檔輸出到 `concept_images/`。

落在六步的第 6 步。這堂對應的鐵律是鐵律一（先切再轉）。

| # | 格 | 標題 | 這張要呈現什麼 | 對應 notebook | 圖 |
|---|---|---|---|---|---|
| 1 | 道 | 問題定位 | 縮放不是每個模型都要做。先判斷下游模型吃不吃距離，再決定要不要動 | `01_scaling_methods.ipynb` | 圖 1 |
| 2 | 術 | 第一原理 | 縮放改的是尺度，不是分佈形狀；要改形狀得用冪次轉換，兩件事不要混為一談 | `02_power_transformations.ipynb` | |
| 3 | 術 | 決策表 | Standard／MinMax／Robust／log／Box-Cox／Yeo-Johnson，比適用情境、對離群值的敏感度、`fit` 學到什麼 | `01_scaling_methods.ipynb` | 圖 2 |
| 4 | 術 | 判斷表 | 依下游模型判斷：樹不用、線性與 KNN 與 SVM 一定要、神經網路通常要 | `01_scaling_methods.ipynb` | |
| 5 | 法 | SOP 五步 | 判斷模型 → 看離群值 → 選方法 → 訓練集 `fit` → 兩邊 `transform` | `04_insurance_case.ipynb` | |
| 6 | 法 | 驗收三問 | 人話、洩漏、拿掉會怎樣。第三問特別有用：對樹模型做縮放，拿掉之後分數不會變，那就是白做的 | — | |
| 7 | 法 | 反例 | 對 XGBoost 做 StandardScaler，分數沒變但欄位變得看不懂了。附前後對照 | `03_outliers_impact.ipynb` | 圖 3 |
| 8 | 器 | 工具速查 | `StandardScaler`、`RobustScaler`、`PowerTransformer` 的最小可跑範例 | `02_power_transformations.ipynb` | |
| 9 | 器 | 三個陷阱與 notebook 對照 | 陷阱：對樹模型做縮放、有離群值還用 MinMax、對測試集重新 `fit` | 全部 | |

要學員記住的那句話：縮放是為了下游模型，不是為了讓數字好看。

對應 notebook 都在 `../../modules/module_05_scaling_transformation/notebooks/` 底下。
