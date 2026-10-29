---
name: decision-analysis
description: 在此專案執行代理特徵、EDA、白盒規則與 ROI 情境分析，或接手既有分析證據。適用商業決策分析，不用於一般教材排版。
---

從 repository 根目錄讀取 `agentic/WORKFLOW.md`，依當前問題選擇需要的環節；操作命令與工具範圍見 `agentic/README.md`。

使用者要示範時，執行 `uv run --locked python -m agentic demo --out <新目錄>`。分析實際資料時，先根據 `agentic/example.config.json` 建立本機設定，核對資料介面與特徵可取得性。

結論須連到執行證據，區分觀測、假設、候選規則與成本情境。接手分析先讀當次 `manifest.json` 與 `handoff.md`，結束前更新交接。
