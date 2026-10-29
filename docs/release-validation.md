# v0.2.0 發行驗證

日期：2026-10-29。此紀錄區分套件、安裝與初始化的實際檢查，以及尚未執行的模型行為驗收。

## 已執行

- `python3 tools/render_skills.py --check`：Claude Code、Codex／OpenCode 共用技能與 OpenCode commands 同步。
- `python3 -m unittest discover -s tests -v`：13 項測試通過，涵蓋發行內容、資源連結、搬移安裝位置、中文／空白路徑、重複初始化、既有文件與指令保留、衝突預檢及符號連結處理。
- `quick_validate.py`：檢查產生的技能入口與發行技能格式。
- `git diff --check`：格式檢查。

以 Node.js v24.16.0、npm 11.13.0 與 skills CLI 1.7.4，在獨立暫存專案實際執行：

```text
skills add <建置後的發行目錄> --skill '*' -a claude-code codex opencode --copy -y
```

未使用全域安裝。安裝器建立九個共用技能與 Claude Code 入口，OpenCode 使用其支援的共用 .agents/skills/。接著從安裝後的 ds-init 執行：

```text
python3 <安裝位置>/ds-init/scripts/init_project.py --root <另一個空白目錄> --opencode-commands
```

成功建立分析骨架與九個 .opencode/commands/*.md，未依賴開發倉庫的 assets 或 references。基本發行內容有 87 個檔案，約 175 KiB；不含教材、真實資料或分析產物。

## 尚未驗證的範圍

未在 Claude Code、Codex、OpenCode 各自呼叫付費模型完成完整分析對話，也未實際開啟 OpenCode TUI 驗證 slash command 的互動。這些需依 [行為情境](../evals/scenarios.md)另行驗收；安裝與腳本通過不代表模型已正確執行研究推論或統計設計。
