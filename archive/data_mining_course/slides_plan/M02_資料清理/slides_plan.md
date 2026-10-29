# 第 2 章　資料清理：投影片規劃

4 張，前情速查性質。不走四格骨架。
產圖提示詞在 [`slides.prompts.md`](slides.prompts.md)。

| # | 標題 | 這張要呈現什麼 | 對應 notebook | 圖 |
|---|---|---|---|---|
| 1 | 資料品質與清理順序 | 清理在整條流程的位置，以及清理順序為什麼不能隨便換 | — | 圖 1 |
| 2 | 型態、重複、分塊速查卡 | 三塊速查：型態轉換常見雷、重複值的三種定義、大檔分塊讀取的參數 | `01_chunking_large_files.ipynb`、`03_data_type_conversion.ipynb` | |
| 3 | 去重的時點 | 同一筆同時落在訓練與測試會發生什麼，這是鐵律一在 M2 的樣子 | `02_handling_duplicates.ipynb` | 圖 2 |
| 4 | 文字清理與 notebook 對照 | 文字欄常見的髒法與處理順序，並列出四本 notebook 的分工 | `04_text_cleaning.ipynb` | |

原本的 Category 記憶體 bytes 對照表和 `.str` 三步驟教學都砍掉，那些查文件就有。

對應 notebook 都在 `../../modules/module_02_data_cleaning/notebooks/` 底下。
