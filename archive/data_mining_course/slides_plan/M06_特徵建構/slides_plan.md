# 第 6 章　特徵建構：投影片規劃

9 張，採 `handout_blueprint.md` 的四格骨架（道 1 · 術 3 · 法 3 · 器 2）。
講義本體 [`M6_Feature_Creation_Fundamentals.md`](../../modules/module_06_feature_creation/slides/M6_Feature_Creation_Fundamentals.md)，產圖提示詞在 [`slides.prompts.md`](slides.prompts.md)，圖檔輸出到 `concept_images/`。

落在六步的第 4 到 5 步。這堂對應的鐵律是鐵律二（特徵要能講人話）。

| # | 格 | 標題 | 這張要呈現什麼 | 對應 notebook | 圖 |
|---|---|---|---|---|---|
| 1 | 道 | 問題定位 | 創造來自真因，不是欄位排列組合。AI 一次生八十個交叉項不算成果 | `01_interaction_features.ipynb` | 圖 1 |
| 2 | 術 | 第一原理 | 每個新欄位都要對得回一條因果鏈；對不回去的，模型就算給它高分也不能留 | `01_interaction_features.ipynb` | |
| 3 | 術 | 決策表 | 交互項／`groupby` 聚合／日期拆欄三類，比什麼時候用、代價、洩漏風險 | `02_group_aggregations.ipynb` | 圖 2 |
| 4 | 術 | 判斷表 | 依欄位型別與因果鏈的位置判斷該造哪一類，並示範怎麼寫下每個欄位的一句理由 | `03_time_derivatives.ipynb` | |
| 5 | 法 | SOP 五步 | 回到因果鏈 → 挑一層 → 造欄位 → 寫下一句話理由 → 檢查有沒有用到全資料 | `04_nyc_taxi_case.ipynb` | |
| 6 | 法 | 驗收三問 | 人話、洩漏、拿掉會怎樣。第一問在這堂是門檻：一句話講不出來就不要留 | — | |
| 7 | 法 | 反例 | 用全資料算品牌平均價再隨機切分，這是聚合特徵最典型的洩漏。附錯誤程式 | `02_group_aggregations.ipynb` | 圖 3 |
| 8 | 器 | 工具速查 | `assign` 造交互項、`groupby().transform()` 造聚合、`dt` 拆日期的最小範例 | `03_time_derivatives.ipynb` | |
| 9 | 器 | 三個陷阱與 notebook 對照 | 陷阱：聚合用了全資料、日期拆完忘了處理週期性、交互項數量爆炸 | 全部 | |

要學員記住的那句話：造得出來不等於該留下來，留下來的每一欄都要能講。

對應 notebook 都在 `../../modules/module_06_feature_creation/notebooks/` 底下。
