# 第 4 章　類別變數編碼：投影片規劃

9 張，採 `handout_blueprint.md` 的四格骨架（道 1 · 術 3 · 法 3 · 器 2）。
講義本體 [`M4_Categorical_Encoding_Handout.md`](../../modules/module_04_categorical_encoding/slides/M4_Categorical_Encoding_Handout.md)，產圖提示詞在 [`slides.prompts.md`](slides.prompts.md)，圖檔輸出到 `concept_images/`。

落在六步的第 6 步。這堂對應的鐵律是鐵律一（先切再轉）。

| # | 格 | 標題 | 這張要呈現什麼 | 對應 notebook | 圖 |
|---|---|---|---|---|---|
| 1 | 道 | 問題定位 | 編碼要同時看理由和洩漏。這堂唯一的新東西是 Target encoding 為什麼危險 | `01_label_onehot_encoding.ipynb` | 圖 1 |
| 2 | 術 | 第一原理 | 編碼是在學一張「類別對到數字」的對照表，那張表就是參數，所以必須只從訓練集學 | `01_label_onehot_encoding.ipynb` | |
| 3 | 術 | 決策表 | Label／One-hot／Count／Target 四種，比適用模型、維度代價、洩漏風險、`fit` 學到什麼 | `03_target_encoding.ipynb` | 圖 2 |
| 4 | 術 | 判斷表 | 兩張判斷表：看類別數量怎麼選、看下游模型是樹還是線性怎麼選 | `04_high_cardinality.ipynb` | |
| 5 | 法 | SOP 五步 | 看基數 → 選方法 → 切分 → 訓練集 `fit` → 對齊測試集欄位 | `05_titanic_case.ipynb` | |
| 6 | 法 | 驗收三問 | 人話、洩漏、拿掉會怎樣。第二問在這堂最關鍵：Target 與 Count 的統計量是不是只從訓練折算的 | — | |
| 7 | 法 | 反例 | 切分前算全資料的類別均值，訓練 AUC 0.97、測試 0.61。附錯誤程式 | `03_target_encoding.ipynb` | 圖 3 |
| 8 | 器 | 工具速查 | `OneHotEncoder(handle_unknown='ignore')`、`TargetEncoder`（sklearn 1.3 起內建 out-of-fold） | `02_count_frequency_encoding.ipynb` | |
| 9 | 器 | 三個陷阱與 notebook 對照 | 陷阱：測試集出現新類別、Target 沒做 out-of-fold、高基數直接 One-hot 爆維度 | 全部 | |

要學員記住的那句話：編碼學到的那張對照表，只能從訓練集長出來。

對應 notebook 都在 `../../modules/module_04_categorical_encoding/notebooks/` 底下。
