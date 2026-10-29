# AI 協作決策分析技能

支援 **Claude Code、Codex、OpenCode**。安裝一次，在自己開的題目資料夾工作；AI 先界定問題並和人類確認，再推導研究計畫、分析與交付。每個題目就是一個專案。

## 安裝與開始

使用 Node.js 22.20 以上安裝技能；初始化只需 Python 3.9 以上：

```bash
npx skills add https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/tree/skills -g
```

選擇你使用的 CLI，勾選全部九個技能。`skills` 分支是輕量發行內容，不包含教材、原始商業資料與分析產物。完整安裝指令見 [安裝指南](docs/distribution-readme.md)。

自己開啟一個分析資料夾，在 CLI 對話使用：

| CLI | 初始化 | 開始研究 |
| --- | --- | --- |
| Codex | `$ds-init 初始化目前專案` | `$ds-frame 我要研究……，資料在……` |
| Claude Code | `/ds-init 初始化目前專案` | `/ds-frame 我要研究……，資料在……` |
| OpenCode | `使用 ds-init skill 初始化目前專案並建立 OpenCode 指令` | `/ds-frame 我要研究……，資料在……` |

OpenCode 首次以自然語言載入 ds-init，初始化時產生 `/ds-*` 指令；其後可以使用 slash command，也可以直接叫用 skill。已有檔案保留，重跑只補缺少部分。

```text
初始化 → AI 推薦研究問題 → 人類確認 → AI 擬定計畫
                                         ↓
                              分析、追問與反證檢查
                                         ↓
                                報告 → 審查 → 交付
```

分析可以反覆迭代，只做 EDA 也可收斂，不強制建模型、提規則或算 ROI。

## 你的分析專案

```text
目前分析專案/
├── research/          問題、計畫、假設、來源、發現與交接
├── data/              raw／interim／processed
├── analysis/          Python、SQL、notebook
├── config/            分組、期間、切分與成本參數
└── reports/           每輪結果、圖表、決策與審查
```

問題與實際確認寫入 research/question.md，研究計畫寫入 plan.md；換工具或對話時讀 handoff.md 接續。原始資料與報告預設不提交，分享前仍需檢查研究文件、程式與設定。分析依賴沿用題目環境，不在初始化時安裝整套套件。

## 文件入口

| 想知道什麼 | 文件 |
| --- | --- |
| 每個技能什麼情境用、怎麼說 | [技能使用指南](.agents/skills/README.md) |
| 安裝與第一個分析專案 | [安裝指南](docs/distribution-readme.md) |
| 問題確認與研究方法 | [研究流程](docs/research-workflow.md) |
| 日常操作與接續 | [操作流程](docs/harness.md) |
| References 與報告骨架 | [技能資源](docs/skill-resources.md) |
| 封裝、初始化與三個 CLI 的設計 | [發行設計](docs/skill-distribution-design.md) |
| 背景、範式轉移與四環節 | [架構白皮書](docs/AI協作決策分析架構.md) |
| 完整文件地圖 | [docs/](docs/README.md) |

## 維護與發行

本分支是開發來源。`skills/` 維護角色模板與方法；產生器將需要的資源一起裝入 `.agents/skills/`（Codex／OpenCode 共用）與 `.claude/skills/`。`.opencode/commands/` 是載入 skill 的指令轉接，不另寫一份分析邏輯。

```bash
python3 tools/render_skills.py --write
python3 tools/render_skills.py --check
python3 -m unittest discover -s tests -v
python3 tools/build_release.py --out /tmp/ds-analysis-release
```

發行目錄需不存在或為空白；只有 SKILL.md、所需 references、模板、初始化腳本、說明與授權。`VERSION` 記錄套件版本，發行分支由建置結果產生。

現行驗證涵蓋產生器、可攜資源、初始化及指令產物；真實模型的研究判斷另外依 [行為情境](evals/scenarios.md)評估，不將檔案測試視為三個 CLI 已完成端到端分析驗收。

開發環境分析套件依 pyproject.toml／uv.lock，以 `uv sync --locked` 安裝。歷史教材見 [archive/](archive/README.md)。
