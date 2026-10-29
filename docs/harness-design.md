# 參考 gstack 的分析協作設計

本專案借鏡 gstack 的角色分工、漸進讀取、共用規約與審查精神，讓 Claude Code／Codex 在目前資料夾處理一個分析題目。新題目由使用者另外開新專案。

下一版將技能安裝位置與分析專案分開，以 ds-init 在使用者自己的資料夾建立骨架；封裝與遷移規格見 [可安裝技能套件設計](skill-distribution-design.md)。此功能尚未實作。

## 保留的設計

| 設計 | 本專案用途 |
| --- | --- |
| 專業角色與路由 | frame、proxy、eda、rules、impact、review、deliver 各負其責 |
| 文件供後續工作接續 | research/ 保存問題、計畫、假設、來源與 handoff |
| 技能單一來源 | skills/ 維護模板，產生 Claude／Codex 入口 |
| 按需讀取 references | 依資料結構與問題選方法，避免載入所有手冊 |
| 報告與審查 | 主要主張連到計算，保留反證、限制與修正紀錄 |

## 讓日常操作保持簡單

使用者直接在對話中描述問題，AI 讀取目前專案文件、確認問題並擬定計畫，再執行與回報。流程以 Markdown 與資料、程式、報告的明確位置維持秩序。

沒有另建多題管理、角色登錄或資料庫調度層；技能產生工具僅供維護者同步入口。分析方法由研究問題決定，不以固定單機台計算流程代表所有任務。

## 品質依據與限制

問題先確認，再推導有來源、有替代解釋與反證條件的計畫。資料切分、特徵可取得性、方法假設及成本口徑，仍由實際程式與審查確認。

文件約定是工作指引，不提供自動權限攔截、身分認證或統計正確性保證。自我審查要標明；讀寫限制依 host 的執行環境。

## 參考來源

本專案以需求取捨，不依賴 gstack 的執行環境。原始參考紀錄來自 2026 年 10 月 29 日查閱的公開文件：

- [gstack README](https://github.com/garrytan/gstack/blob/main/README.md)：角色與協作流程。
- [gstack ARCHITECTURE](https://github.com/garrytan/gstack/blob/main/ARCHITECTURE.md)：技能產生與架構取捨。
- [gstack hosts](https://github.com/garrytan/gstack/blob/main/docs/hosts.md)：不同 host 的適配。

研究推論依據見 [研究設計 reference](../skills/ds-frame/references/research-design.md)，實際使用見 [技能操作](harness.md)。
