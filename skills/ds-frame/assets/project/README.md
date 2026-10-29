# 分析專案

本目錄是一個獨立分析題目。先定義問題並請使用者確認，再擬定研究計畫與執行分析。

| 位置 | 用途 |
| --- | --- |
| `research/question.md` | 研究問題、決策範圍與待確認事項 |
| `research/plan.md` | 確認問題後的研究假設、替代解釋、方法與執行順序 |
| `research/hypotheses.md` | 計畫使用的假設、反證與預定判準 |
| `research/findings.md` | 每輪假設狀態與證據，不回填改寫原定義 |
| `research/sources.md` | 文獻、領域資料與方法來源 |
| `research/confirmations/` | 使用者對具體問題版本的確認來源 |
| `data/raw/` | 原始資料快照，保留原檔並記錄來源 |
| `data/interim/` | 清理與對齊過程的中間資料 |
| `data/processed/` | 可供分析的資料，保留切分與轉換方式 |
| `analysis/` | 本題的 Python、查詢與 notebook |
| `config/` | 分組、期間、成本與分析參數；不存憑證 |
| `reports/explorations/` | 每輪獨立結果；建議以 round-001 等子目錄保存 |
| `reports/figures/`、`reports/tables/` | 報告採用的圖表與統計表 |
| `reports/decision.md` | 決策報告草稿 |
| `artifacts/`、`state.sqlite` | Harness 管理的快照與狀態，不手動修改 |
| `handoff.md` | 接手所需的進度、來源與下一步 |

所有命令從 repository 根目錄執行，指定本專案的 `--case`。本目錄中的研究文件是空白骨架，尚未登錄為證據或取得人類確認。
