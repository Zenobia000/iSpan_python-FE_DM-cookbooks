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

## 要重跑的話

它讀三個檔案，全部用相對路徑，且都在**上一層** `projects/project/`：

| 讀取路徑 | 實際位置 |
| :--- | :--- |
| `Automobile_data.csv` | `../Automobile_data.csv`（UCI Automobile，205 列） |
| `car_data/audi.csv` | `../car_data/audi.csv` |
| `car_data/ford.csv` | `../car_data/ford.csv` |

所以直接在 `legacy/` 開會找不到檔案——Jupyter 的工作目錄是 notebook 自己的目錄。
要重跑就把它複製回上一層再開，或在第一格加 `%cd ..`。

不過多數情況不需要重跑：26 個 code cell 的輸出都還在，GitHub 的 notebook 預覽
直接看得到那些 Altair 圖表，檔案 11 MB 也大多是這些內嵌輸出。
