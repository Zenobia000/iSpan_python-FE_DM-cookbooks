# 文件地圖

本專案以 Claude Code／Codex 原生技能與本機證據管理，支援可續作的決策分析。先理解使用流程，再視需要閱讀方法與開發細節。

## 從哪裡開始

| 想了解的問題 | 文件 |
| --- | --- |
| 怎麼開始、有哪些資料夾？ | [專案首頁](../README.md) |
| 新題目如何確認問題並建立獨立目錄？ | [研究流程與專案工作區](research-workflow.md) |
| 每個技能什麼時候用、怎麼提問？ | [技能使用指南](../.agents/skills/README.md) |
| 怎麼用技能、建立 case 與交付？ | [Harness 操作](harness.md) |
| 為什麼這樣分工、借鏡了哪些 gstack 設計？ | [架構與設計取捨](harness-design.md) |
| References 與報告骨架何時使用？ | [技能參考與模板](skill-resources.md) |
| 每個角色必須交付什麼證據？ | [證據契約](harness-contracts.md) |
| AI 協作決策分析的背景與四個環節？ | [架構白皮書](AI協作決策分析架構.md) |
| 陌生領域、特徵工程與選圖的方法？ | [方法參考](references/README.md) |
| 如何驗證真實 agent 行為？ | [行為評估情境](../evals/scenarios.md) |
| 想看可以執行的分析？ | [單一機台計算範例](../examples/factory_analysis/README.md) |
| 舊課程與原始素材在哪裡？ | [封存索引](../archive/README.md) |

## 文件與實作的關係

`harness.md` 與 `harness-contracts.md` 描述目前的使用方式和契約。白皮書說明方法與參考設計，其中的 `.harness/`、`stat_profiler.md` 等配置是概念示例；操作本專案時使用 `skills/ds-*`、`tools/ds_harness.py` 與 `projects/<題目>/`；舊 `.decision/cases/` 可原路徑續作。

目前已實作角色路由、雙 host 技能入口、獨立專案骨架、問題版本確認、研究計畫、輸入與證據版本、失效檢查、review 與本機交付快照。探索工作台、每輪問題管理、runtime 自動選載方法 references 與互動 HTML 報告尚非現有功能；方法參考與產出模板已由技能規則按需引用。

## 維護位置

- 技能內容改 `skills/<角色>/SKILL.md.tmpl`；再以 `tools/render_skills.py` 產生 `.agents/` 與 `.claude/` 入口。
- 執行與版本管理改 `tools/`，以 `tests/` 驗證。
- 共用操作與設計說明放 `docs/`；方法來源放 `docs/references/`。
- 真實資料與 case 放在本機 `projects/<題目>/`；既有 `.decision/` 可原路徑續作，不提交。
- `archive/` 保存歷史內容，平常不參與技能載入與 Harness 測試。
