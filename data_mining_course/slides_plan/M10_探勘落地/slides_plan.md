# 模組10 資料探勘應用：投影片規劃

9 張，走 `handout_blueprint.md` 的四格骨架（道 1 · 術 3 · 法 3 · 器 2）。
講義本體 [`M10_Data_Mining_Applications_Fundamentals.md`](../../modules/module_10_data_mining_applications/slides/M10_Data_Mining_Applications_Fundamentals.md)，產圖提示詞在 [`slides.prompts.md`](slides.prompts.md)，圖檔輸出到 `concept_images/`。

落在六步的第 6（驗收與落地） 步。這堂對應的鐵律是鐵律二（特徵要能講人話）。

| # | 格 | 標題 | 這張要呈現什麼 | 對應 notebook | 圖 |
|---|---|---|---|---|---|
| 1 | 道 | 這堂在解什麼 | 這堂把前面所有工具接成一條線，重點不是演算法步驟，是重要性要對得回理由 | `03_tree_models/01_xgboost_features.ipynb` | 圖 1 |
| 2 | 術 | 第一原理 | 樹模型的重要性反映的是「在這棵樹裡好不好切」，不是因果強度 | `03_tree_models/02_lightgbm_features.ipynb` | |
| 3 | 術 | 決策表 | 樹重要性／分群／關聯規則三類落地方式，比適用問題、參數敏感度、結果怎麼驗收 | `02_clustering/01_kmeans_clustering.ipynb` | 圖 2 |
| 4 | 術 | 判斷表 | 依問題型態判斷：要預測用樹、要分群用 K-Means 或 DBSCAN、要看共現才用關聯規則 | `02_clustering/02_dbscan_clustering.ipynb` | |
| 5 | 法 | SOP 五步 | 端到端 `Pipeline` 五步：前處理進 Pipeline → 切分 → 訓練 → 看重要性 → 對回 Why | `04_end_to_end_pipeline.ipynb` | |
| 6 | 法 | 驗收三問 | 人話、洩漏、拿掉會怎樣。第一問是這堂的驗收核心，排名前五的特徵都要能口頭講一遍 | — | |
| 7 | 法 | 反例 | `customerID` 編碼後排名第一還照用，模型記住的是流水號不是行為。附錯誤輸出 | `03_tree_models/03_telco_churn_case.ipynb` | 圖 3 |
| 8 | 器 | API 速查 | `XGBoost`、`LightGBM` 的重要性取用、`KMeans`、`DBSCAN` 的最小範例 | `01_association_rules/01_apriori_algorithm.ipynb` | |
| 9 | 器 | 三個陷阱與 notebook 對照 | 陷阱：把編碼後的 ID 當特徵、分群前沒縮放、把 Apriori 當現代推薦主線 | 全部 | |

要學員記住的那句話：模型說重要，你要說得出為什麼重要。

對應 notebook 都在 `../../modules/module_10_data_mining_applications/notebooks/` 底下。
