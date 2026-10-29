# 決策分析角色的證據契約

此處定義 agent 需要交付的內容。各 stage 文件可使用 Markdown、JSON 或文字；runtime 驗證版本與相依性，內容品質交由 ds-review。

| Stage | 必要上游 | 文件需要回答的問題 |
| --- | --- | --- |
| frame | 無 | 誰要做什麼決定 指標與比較基準 觀測單位 時間 影響門檻 已知限制與分析計畫 |
| plan | 已確認的 frame | 既有證據與缺口 假設及來源 替代解釋 反證條件 推薦順序 方法與驗證 產出位置 |
| proxy | frame plan | 缺少什麼概念 用哪些訊號替代 為何合理 計算與單位 當下可取得性 混雜因素與驗證方式 |
| eda | frame plan | 資料品質 方法與假設 效果量 樣本與不確定性 多重比較 有用與不成立的發現 重現命令 |
| rules | frame plan eda | 條件與單位 適用範圍 獨立驗證 涵蓋率與錯誤成本 現場處置與試行退出條件 |
| impact | frame plan eda | 觀測與假設的來源 成本期間 公式 情境 敏感度 損益兩平與試行評估 |

frame 由 ds-frame 登錄並取得使用者對目前版本的確認；plan 同樣由 ds-frame 在確認後登錄。確認來源與計畫流程見 [research-workflow.md](research-workflow.md)。

方法參考與產出骨架見 [skill-resources.md](skill-resources.md)。模板幫助保留必要內容；runtime 不以章節存在判定統計、領域或成本推論正確。

proxy 與 rules 可省略。若存在任一上游，其版本及有效性就會納入後續 stage；runtime 採保守失效策略。

每項分析文件連到其實際資料、Schema、程式／查詢與計算輸出，並以 `record --input` 登錄這些檔案。僅寫「檢查通過」不能替代可重現的證據。沒有資料的假設提案可以登錄，但必須保留待驗證狀態。

review 檢查 scope、leakage、statistics、business，並閱讀交付草稿。delivery 包含當前證據版本、review_id、已審查的報告 SHA-256 與本機快照；不表示已執行業務行動。所有紀錄可供另一個 host 接手。
