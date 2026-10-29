@AGENTS.md

Claude Code 與 Codex 共用根目錄協作判準。決策分析使用 `decision-analysis` 路由或指定的 `ds-*` skill；跨 agent 續作時先對本機 case 執行 `python3 tools/ds_harness.py status --case <case>`。

新題目以 `projects/<題目>/` 建立獨立資料、程式與報告目錄。先由 ds-frame 定義研究問題、取得使用者確認，再擬定 plan 並執行分析；流程見 `docs/research-workflow.md`。
