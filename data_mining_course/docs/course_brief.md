# 課程 Brief：章節與內容

> 一頁看完整門課。細節：心法 [`M0_心法總綱.md`](M0_心法總綱.md)、講義規格 [`handout_blueprint.md`](handout_blueprint.md)、宗旨 [`course_charter.md`](course_charter.md)。

**這門課在補什麼：** 學員已會 ML、pandas、sklearn，手上有 AI 工具。
會寫 API 不是瓶頸——這門課補的是**丟進模型前，資料要變成什麼、為什麼**，
以及 AI 寫出轉換之後，人要盯什麼。

**三條鐵律（全課只有這三條，其餘都是實例）**

1. **先切再轉** — 所有 `fit()` 只看訓練集／訓練摺；時序測試集必須較晚。
2. **特徵要能講人話** — 對不回 Why 的高分特徵，先當可疑。
3. **資料形狀決定能接哪類模型** — 張量 shape 與標籤格式先設計，再挑模型。

---

## 章節

| 章 | 主軸 | 技法 | 資料 | 人必須盯 |
|---|---|---|---|---|
| **M0** 心法總綱 | 三條鐵律 + 六步 | SCQA、5 Why（含逆推驗證）、KT IS／IS NOT、SMART | — | 因果鏈逆推唸不唸得通 |
| **M1** EDA | 系統化掃描，不是漫無目的畫圖 | `info`／`describe`、分佈與相關性、選圖決策 | Titanic | 問題有沒有寫清楚 |
| **M2** 清理 | GIGO | 分塊讀取、重複值、型態轉換、文字清理 | — | 切分**前**去重 |
| **M3** 缺值與異常 | 缺值可以是訊號 | MCAR／MAR／MNAR、插補、IQR／Z-score | House Prices | 中位數只能從訓練集算 |
| **M4** 類別編碼 | 編碼看 Why 與洩漏 | Label／One-hot／Count／Target、高基數 | Titanic | Target／Count 只能看訓練摺 |
| **M5** 縮放與轉換 | 樹通常不用縮放 | Standard／MinMax／Robust、log／Box-Cox／Yeo-Johnson | Insurance | 線性／KNN 才要；參數看訓練集 |
| **M6** 特徵創造 | 創造來自真因，不是欄位排列組合 | 交互、`groupby` 聚合、日曆拆欄 | NYC Taxi | 留下的欄要能一句話解釋 |
| **M7** 選擇與降維 | 重要性對不回 Why 就可疑 | 過濾／包裹／嵌入、PCA | Breast Cancer | 不要用包裹法尋寶 |
| **M8** 時間序列 | 按時間切，不准洗牌 | `shift` lag、`rolling`／`expanding`、季節分解 | 電力消耗 | 測試集必須是較晚的一段 |
| **M10** 探勘落地 | 高重要性要能講人話 | 樹重要性（XGB／LGBM）、端到端 `Pipeline`、分群；關聯規則一小段 | Telco、Mall | `customerID` 變重要＝無意義編碼 |
| **M9** 多模態前處理 | 換模態不換腦袋：理解 → 清理 → 導入張量 | tokenizer、ViT／CLIP、Whisper、VideoMAE、圖文配對 | IMDB／貓狗／UrbanSound／影片 | 說得出這個嵌入代表什麼 |
| **M11** 下游訓練 | 資料形狀 → 能接哪類模型 | `Trainer`、LoRA／QLoRA、SFT、RAG、VLM 藍圖 | 真實小樣本 | 這是最小 demo，不是從零訓練大模型 |
| **總整** | 用同一套六步走完一輪 | 5 Why → SMART → 先切再轉 → 可解釋驗收 | 英國二手車 | Why、洩漏、口頭辯護 |

---

## 上課順序

```
M0 心法（開場 40 分）→ M1–M2 前情（自學）→ M3–M8 表格工具
→ M10 探勘落地 → M9 + M11 AI 前處理 → 總整（二手車）
```

**七週建議進度：** ① M1 ② M2+M3 ③ M4+M5 ④ M6+M7 ⑤ M8 ⑥ M10 ⑦ M9+M11 → 總整
（先探勘落地再進 AI 前處理；M9／M11 連著上，資料形狀那條線不切斷。）

**兩條線不混一本：** 表格線（M1–M8、M10，假設從 domain 來）／大模型線（M9、M11，假設從資料結構來）。共用三條鐵律，不共用 notebook。

## 六步（技法只出現在第 5 步之後）

`要決定什麼 → 5 Why 挖真因並逆推驗證 → SMART 篩 → domain 說明 → 資料對不對得上 → 才選技法`

驗收三問：**人話？洩漏？拿掉它模型會怎樣？**

## AI 契約

AI 提出假設、寫轉換程式；**人負責 Why、SMART、先切再轉、洩漏**。
AI 一次生 80 個交叉項不算成果。

---

## 教材對照

| 要什麼 | 看哪裡 |
|---|---|
| 一頁總覽（章節與內容） | 本檔 |
| 心法本體（第 0 章講義） | `docs/M0_心法總綱.md` |
| 課程宗旨與涵蓋範圍 | `docs/course_charter.md` |
| 講義撰寫規格（道術法器四格） | `docs/handout_blueprint.md` |
| 上課主線 | `modules/module_01`–`11`（各模組 `slides/` 是講義、`notebooks/` 是實作） |
| 最後總整 | `capstone/capstone_used_car.ipynb` |
| 環境與路徑問題 | `docs/faq.md` |
| 複本／大 CSV | `modules/extension/` |
