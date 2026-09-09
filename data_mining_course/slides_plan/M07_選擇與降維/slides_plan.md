# 第 7 章　特徵選擇與降維：投影片規劃

9 張，採 `handout_blueprint.md` 的四格骨架（道 1 · 術 3 · 法 3 · 器 2）。
講義本體 [`M7_Feature_Selection_and_Reduction.md`](../../modules/module_07_feature_selection/slides/M7_Feature_Selection_and_Reduction.md)，產圖提示詞在 [`slides.prompts.md`](slides.prompts.md)，圖檔輸出到 `concept_images/`。

落在六步的第 6（驗收） 步。這堂對應的鐵律是鐵律二（特徵要能講人話）。

| # | 格 | 標題 | 這張要呈現什麼 | 對應 notebook | 圖 |
|---|---|---|---|---|---|
| 1 | 道 | 問題定位 | 重要性是模型的意見，不是事實。對不回理由的高分特徵，先當可疑 | `03_embedded_methods.ipynb` | 圖 1 |
| 2 | 術 | 第一原理 | 過濾看的是特徵與目標的關係，包裹與嵌入看的是「在這個模型裡有沒有用」，三者答案本來就會不一樣 | `01_filter_methods.ipynb` | |
| 3 | 術 | 決策表 | 過濾／包裹／嵌入／PCA 四類，比計算成本、會不會受模型影響、洩漏風險 | `02_wrapper_methods.ipynb` | 圖 2 |
| 4 | 術 | 判斷表 | 依欄位數量與時間預算判斷用哪類；為什麼不要拿包裹法當尋寶工具 | `04_dimensionality_reduction.ipynb` | |
| 5 | 法 | SOP 五步 | 先砍明顯無用 → 過濾初篩 → 嵌入法排序 → 對回 Why → 放進 `Pipeline` 每折重做 | `05_breast_cancer_case.ipynb` | |
| 6 | 法 | 驗收三問 | 人話、洩漏、拿掉會怎樣。第一問決定去留：排名很前面但講不出理由的欄位，先查是不是洩漏或無意義編碼 | — | |
| 7 | 法 | 反例 | 在切分前用全資料選完欄位才做交叉驗證，分數灌水。附錯誤程式與正確寫法 | `02_wrapper_methods.ipynb` | 圖 3 |
| 8 | 器 | 工具速查 | `SelectKBest`、`RFE`、`feature_importances_`、`PCA` 的最小可跑範例 | `03_embedded_methods.ipynb` | |
| 9 | 器 | 三個陷阱與 notebook 對照 | 陷阱：選擇沒放進 `Pipeline`、把 PCA 當特徵選擇、只信一種重要性指標 | 全部 | |

要學員記住的那句話：重要性排名是線索，不是結論。

對應 notebook 都在 `../../modules/module_07_feature_selection/notebooks/` 底下。
