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

## GitHub 發行後檢查

從公開輕量分支實際執行同樣的安裝流程：

```text
skills add https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/tree/skills --skill '*' -a claude-code codex opencode --copy -y
```

成功安裝全部九個技能至獨立暫存專案，包含 Claude Code 與 Codex／OpenCode 的共用入口。從 GitHub 下載的 ds-init 腳本可在另一個目錄建立研究骨架及九個 OpenCode 指令；再次執行時 created 與 updated 均為空。

發行分支提交為 `7df9637`，對應實作提交 `7fb5faf`。後者的 [GitHub Actions](https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/actions/runs/40873952537) 已完成且成功，包含 Python 3.9／3.12 的產生器同步與測試。

## 尚未驗證的範圍

未在 Claude Code、Codex、OpenCode 各自呼叫付費模型完成完整分析對話，也未實際開啟 OpenCode TUI 驗證 slash command 的互動。這些需依 [行為情境](../evals/scenarios.md)另行驗收；安裝與腳本通過不代表模型已正確執行研究推論或統計設計。

## v0.2.1：Hypothread 品牌與發行圖文

新增 Hypothread｜問脈品牌、透明 icon、主視覺、SVG 流程圖、繁體中文及英文 README、推廣文案與 release 說明。圖片由內建 image_gen 產生，未指定或宣稱底層模型版本；生成 prompts 與使用方式見 [品牌定義](marketing/brand.md)。

本次實際執行產生器同步檢查、13 項測試與 git diff --check，全部通過。發行測試新增圖片與文字的逐位元組比對，並檢查 README 圖片的相對路徑，避免發行分支遺漏素材。

以本機 Chrome 檢視 README 的桌面與手機閱讀預覽，確認品牌圖片、安裝內容與表格可呈現。這是 GitHub 風格的本機閱讀預覽，並非 OpenCode 或完整模型分析的互動驗收。

基本技能核心維持原有規模；含品牌圖片及雙語 README 的 v0.2.1 發行目錄共 91 個檔案，約 1.75 MiB。技能功能與既有 ds-* 指令保持相容。
