# 上課地圖（2026）

對齊 [course_charter.md](course_charter.md)。講義是 overview；程式一本一份真實資料。

對象已會 ML，且手上有 AI 工具。每堂：**AI 可寫轉換；人盯 Why、切分、洩漏。**

| 模組 | 一句主軸 | 資料主角 | AI 可以 | 人必須盯 |
|---|---|---|---|---|
| 1–2 | 前情複習 | Titanic（只在 M1 掃描） | 畫圖、Pandas 語法 | 問題有沒有寫清楚 |
| 3 | 缺值可以是訊號 | House Prices | 插補程式 | 參數只看訓練集；缺本身是不是資訊 |
| 4 | 編碼看 Why 與洩漏 | Titanic 編碼繞回 | Label／One-hot 程式 | Target／Count 只能看訓練摺 |
| 5 | 樹通常不用縮放 | Insurance | Standard／MinMax | 線性／KNN 才要；參數看訓練集 |
| 6 | 創造來自真因 | NYC Taxi | groupby、日曆拆欄 | 不要排列組合；日曆本堂講完 |
| 7 | 重要性對不回 Why 就可疑 | Breast Cancer | 過濾法 API | 不要用包裹法當尋寶 |
| 8 | lag／rolling；按時間切 | 電力消耗（真實檔） | 窗特徵程式 | 測試集必須較晚 |
| 9 | 理解 → 清理 → 導入張量 | IMDB／貓狗／UrbanSound／影片片段 | tokenizer、processor | 嵌入代表什麼；不要當黑盒特徵本體 |
| 10 | 落地應用 | Telco、Mall；Instacart 選做 | 樹模型 fit | 關聯規則不是現代推薦主線；高重要性要能講人話 |
| 11 | 資料形狀決定能接哪類模型 | 真實小樣本 | Trainer／LoRA 骨架 | 這堂是導入與最小 demo，不是從零訓練大模型 |
| 總整 | 心法走一遍 | 英國二手車 | 提假設、寫轉換 | Why、SMART、洩漏、口頭辯護 |

總整：[`capstone/capstone_used_car.ipynb`](../capstone/capstone_used_car.ipynb)
