# 分析專案工作區

一個新題目建立一個資料夾，各自保存資料、研究計畫、程式、報告與狀態。例如 `projects/factory-quality/` 與 `projects/customer-retention/` 是兩個互不共用工作檔的專案。

在 repository 根目錄執行：

```bash
python3 tools/ds_harness.py init --case projects/factory-quality --goal eda
python3 tools/ds_harness.py init --case projects/customer-retention --goal impact
```

工具自動建立 research/、data/raw/、data/interim/、data/processed/、analysis/、config/、reports/、artifacts/、state.sqlite 與交接骨架。已有專案使用 status 續作，init 不覆寫既有目錄。

專案先完成研究問題與人類確認，再由 AI 推導並登錄研究計畫。完整流程與資料夾用途見 [研究工作流程](../docs/research-workflow.md)。

此目錄只追蹤本說明，個別專案內容預設忽略。這是檔案與狀態管理層的隔離；讀寫權限仍由實際執行環境控制。
