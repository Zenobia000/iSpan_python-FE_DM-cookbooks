# 舊版：對照用，不是教材

`awosome_visualiztion_v2.ipynb` 是二手車分析的 **Altair 舊 V2**，保留下來當對照組。

現行路徑：

- 市場 EDA（上架價、折舊、車型）：上一層 `car_market_eda.ipynb`
- 規格表主體 + 品牌弱對齊：上一層 `awesome_visualization_v2.ipynb`（seaborn + plotly）

要學就看那兩本。這本只用來看「改了什麼」。

## 三版的差別

| | 舊 V2（本檔） | `car_market_eda.ipynb` | 新 V2 `awesome_visualization_v2.ipynb` |
| :--- | :--- | :--- | :--- |
| 主表 | Automobile 先走，再接 Audi／Ford 上架檔 | `car_data/` 九檔 | `Automobile_data.csv` 205 列 |
| 敘事 | 逐個變數看過去 | 全貌先行 → 五題深挖；第 6.1 節看車型 | 規格／保險欄怎麼拉開目錄價，最後才弱對齊 |
| 視覺化 | Altair 為主 | matplotlib + seaborn | seaborn + plotly（沒裝 plotly 會退回 seaborn） |
| 資料 | 含 `cclass` / `focus` | 排除子集檔；不分析 Automobile | 先清 `?`；英國檔只在最後一節聚合成品牌列 |
| 對齊 | 沒有講清楚兩表不能併 | 兩來源角色分開 | `make` → `brand`，五個重疊品牌、有標價約 66 列 |

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
