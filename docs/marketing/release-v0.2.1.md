# Hypothread｜問脈 — From questions to evidence.

把問題想清楚，讓分析有據可循。

Hypothread 是一套給 Claude Code、Codex 與 OpenCode 的資料科學協作技能。AI 先釐清研究問題、取得人類確認，再沿著假設、替代解釋與反證推進分析，將證據和交接留在使用者自己的專案裡。

## 這個版本帶來什麼

- 完整的產品名稱、透明 icon、研究筆記風格主視覺與流程圖。
- 重寫的繁體中文 README 與英文介紹，包含示意探勘、使用情境及第一次分析的操作方式。
- 九個可安裝技能，從 ds-init、ds-frame、ds-eda 開始，按需要使用代理特徵、規則、效益、審查與交付角色。
- 隨套件附上的方法 references、產出模板與 OpenCode 指令轉接；現有 ds-* 指令保持相容。

## 安裝

```bash
npx skills add https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/tree/skills -g
```

選擇你的 CLI，勾選全部九個技能。Claude Code 使用 `/ds-init`，Codex 使用 `$ds-init`；OpenCode 首次說「使用 ds-init skill 初始化目前專案，並建立 OpenCode 指令」。之後以 ds-frame 定義研究問題。

需要 Node.js 22.20 以上與 Python 3.9 以上。模型與運算費用依你自己的環境計算。

發行附件提供完整套件 ZIP、透明 icon 與主視覺；輕量 skills 分支持續提供標準安裝器入口。

## 驗證與適用範圍

13 項測試涵蓋初始化、既有檔案保留、可攜資源、指令轉接與發行圖片完整性。README 的案例明列為示意，不是實測分析結果。三個 CLI 的完整模型分析對話及 OpenCode TUI 互動仍需更多實際驗收。

歡迎在 Issues 分享你想研究的一個問題，以及使用過程卡住的步驟。

---

Hypothread brings research-first data science skills to Claude Code, Codex, and OpenCode: frame the question, confirm its scope, develop hypotheses and alternatives, then keep exploring with evidence saved in your own project.

This release adds the Hypothread identity, an illustrated README, an English introduction, and complete release artwork. All nine skill names remain compatible. Bundled instructions are primarily in Traditional Chinese. See README.en.md for the English guide; installation and workspace behavior are tested, while full model-driven analysis validation is ongoing.
