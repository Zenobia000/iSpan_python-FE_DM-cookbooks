# 任務適配契約

由 research 盤點，其他技能在各自職責內維護。每個欄位標示已核對／假設／缺失及來源日期。

| 契約 | 最少內容 |
| --- | --- |
| 任務 | 題型、分析單位、目標、官方 metric 與方向、交付類型、期限、規則 URL |
| 資料 | 來源／版本／授權、ID、label、可用時點、外部資料及模型資產限制 |
| 執行 | 本機或平台、依賴版本、網路、CPU/GPU、記憶體、時間與試驗預算 |
| 驗證 | split IDs、群組／時間邊界、metric 實作、基準與最終測試保留方式 |
| 實驗 | run ID、父 run、實際設定、資料／程式版本、seed、命令、狀態與產物識別 |
| 交付 | CSV／notebook／package／環境介面、schema、推論入口、載入與平台測試 |

ready：所需契約與 adapter 齊備，可開始小規模執行。needs-adapter：可規劃但缺 loader、metric 或交付介面。blocked：權限、規則或資源阻止目前執行。狀態不是永久支援宣告。

Kaggle 不預設允許外部資料、網路或固定 GPU／每日提交額度。查閱該題最新規則；下載、接受規則、提交與發布依當次授權範圍執行。Public leaderboard 不取代本地驗證。

官方參考：[Competitions](https://www.kaggle.com/docs/competitions)、[Packages](https://www.kaggle.com/docs/packages)、[CLI](https://github.com/Kaggle/kaggle-cli)。通用契約不等於已完成任一競賽的實跑驗收。
