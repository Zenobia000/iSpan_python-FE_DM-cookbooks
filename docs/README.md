# 文件地圖

目前資料夾就是一個分析專案；新題目由使用者另開資料夾。從 [專案入口](../README.md) 開始，直接在對話中叫用技能。

| 想了解什麼 | 閱讀文件 |
| --- | --- |
| 每個 skill 何時用、怎麼說 | [技能使用指南](../.agents/skills/README.md) |
| 日常操作與接續分析 | [操作流程](harness.md) |
| 問題確認、研究推論與目錄 | [研究流程](research-workflow.md) |
| 各角色需要什麼證據 | [角色契約](harness-contracts.md) |
| 為何採用技能與檔案交接 | [架構設計](harness-design.md) |
| 下一版如何安裝與初始化自己的專案 | [可安裝技能套件設計（規格）](skill-distribution-design.md) |
| references 與報告骨架 | [資源索引](skill-resources.md) |
| 背景、範式轉移與四環節 | [架構白皮書](AI協作決策分析架構.md) |
| 特徵工程方法來源 | [參考資料](references/README.md) |

`skills/` 保存技能原稿與方法資源；`.agents/skills/`、`.claude/skills/` 是兩個工具的產生入口。研究文件放 `research/`，資料放 `data/`，程式放 `analysis/`，參數放 `config/`，報告放 `reports/`。

歷史教材位於 [archive](../archive/README.md)，不參與日常分析流程。原始商業資料、憑證與執行產出不提交；分享前檢查研究文件與設定是否含敏感內容。
