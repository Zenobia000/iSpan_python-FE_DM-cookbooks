# 舊版：對照用，不是教材

`awosome_visualiztion_v2.ipynb` 是二手車分析的**前一版**，保留下來當對照組，
現行教材是上一層的 `car_market_eda.ipynb`。要學就看那本，這本只用來看「改了什麼」。

## 兩版的差別

| | 舊版（本檔） | 現行 `car_market_eda.ipynb` |
| :--- | :--- | :--- |
| 敘事結構 | 逐個變數看過去 | 全貌先行 → 問題浮現 → 針對性深挖 |
| 視覺化 | Altair 為主 | matplotlib + seaborn |
| 資料整合 | 含 `cclass` / `focus` | 排除兩者——它們是 merc / ford 的子集，併入會重複計算 |
| 選圖說明 | 無 | 每章附「問題型態 → 圖表」的推理 |

改版的動機寫在 commit `35f0d54` 的訊息裡。

## 這本跑不起來

第一個 cell 讀 `Automobile_data.csv`（UCI Automobile 資料集），那份資料不在這個 repo，
`car_data/` 只有英國二手車的 13 個檔案。保留的是**輸出**——26 個 code cell 的圖表都還在，
直接看 GitHub 的 notebook 預覽即可，不需要重跑。檔案 11 MB 也大多是這些內嵌圖表。
