# 參考 gstack 的決策分析 Harness 設計

上一版以 `agentic/` 包裝固定的 Python 計算流程，並用一個 skill 呼叫它。這讓工具介面決定了資料型態、分析方法與交付順序，與原本希望 AI 協助突破假設限制的目的不一致。

新版將領域技能、任務狀態、證據驗收與計算工具分開。四個核心環節保留為專業角色，讓 agent 在 Claude Code／Codex 中依業務問題決定要做哪些工作，計算程式則是可替換的工具。

## gstack 的參考與本專案的取捨

| gstack 設計 | 本專案採用方式 |
| --- | --- |
| 專業角色技能與任務路由 | frame proxy eda rules impact review deliver 與路由入口 |
| 一個技能產出的文件供後續技能使用 | 具版本的 stage 快照與相依性 |
| 模板產生 host 對應技能 產物提交版控 | `skills/` 單一來源 產生 Claude／Codex 入口 CI 檢查漂移 |
| 共用啟動與完成協定 | 輕量共用 session 契約 與明確完成／缺上下文狀態 |
| review 起點與候選版本綁定 | review-start 擷取候選與報告 review-finish／deliver 檢查版本 |
| 可供 agent 自行修復的錯誤 | JSON 錯誤含 code reason recovery |
| 免費確定性檢查與付費真實 agent eval 分層 | stdlib runtime 測試與未執行的 host 行為 eval 規準分開 |

這是依領域需求重新實作，不是安裝或 vendoring gstack。瀏覽器 daemon、Bun、merge train、Capy Captain、遙測與全域自動更新沒有這次分析任務的必要性，因此未引入。

## 重要邊界

技能文件引導 agent 如何判斷；runtime 只強制可以機械檢查的條件。把「不要洩漏」寫在指引中，無法等同真正通過洩漏檢查；把 session 宣告為另一個 host，也無法證明已執行跨模型審查。報告需清楚區分這些證據層級。

最小可用路徑為 frame → eda → 草稿 → review → delivery。只有缺少直接訊號才做 proxy，需要現場操作條件才做 rules，需要效益比較才做 impact。對未達資料條件的專案，可以交付清楚的限制與追加量測方案。

## 參考來源

於 2026 年 10 月 29 日參考下列 gstack 原始碼與文件。外部主分支會持續更新；本專案不在執行時依賴該主分支。

- [README](https://github.com/garrytan/gstack/blob/main/README.md)：角色技能、路由與工作流程。
- [ARCHITECTURE](https://github.com/garrytan/gstack/blob/main/ARCHITECTURE.md)：模板系統、可操作錯誤與 eval 分層；參考檔案 blob SHA `7dc9c3546e6ce00c099529ef8c77e88ca643e82d`。
- [gen-skill-docs.ts](https://github.com/garrytan/gstack/blob/main/scripts/gen-skill-docs.ts)：單一模板與不同 host 的產物。
- [review template](https://github.com/garrytan/gstack/blob/main/review/SKILL.md.tmpl)：審查起點、候選綁定與修正後重審；參考檔案 blob SHA `b22eaf4425000fdf3713679a8b6324089ffdd362`。
- [hosts](https://github.com/garrytan/gstack/blob/main/docs/hosts.md)與[recommended harness](https://github.com/garrytan/gstack/blob/main/docs/recommended-harness.md)：host 適配與平台專屬能力的區分。
- [Codex Skills](https://learn.chatgpt.com/docs/build-skills)與[Claude Code Skills](https://code.claude.com/docs/en/skills)：本 repository 的原生技能入口。
