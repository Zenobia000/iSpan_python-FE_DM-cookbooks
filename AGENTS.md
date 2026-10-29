# 專案工作入口

使用繁體中文溝通，程式碼註解使用英文。保留既有未提交工作，不自動刪除教材、改寫原始資料、提交或推送。

## 科學分析 Harness

表格資料的問題釐清、視覺探索、特徵工程、模型比較或接續研究，先讀 [.agents/skills/scientific-analysis/SKILL.md](.agents/skills/scientific-analysis/SKILL.md)。
這是單一 skill 配本機腳本的研究助手，不是 MCP server 或常駐分析網站。方法參考按需載入，不預先讀取全部教材。

以 `pyproject.toml`／`uv.lock` 管理專案環境，不安裝系統語言套件。原始資料唯讀，新分析產物寫到新 run。
先解釋資料與模型行為，再比較性能；保留可解釋基準，不把模型重要性或相關性當成因果。

舊課程技術範圍見 [docs/legacy-curriculum-outline.md](docs/legacy-curriculum-outline.md)。先前架構草案屬設計歷程，實際可用能力以 skill、CLI 及已執行測試為準。
