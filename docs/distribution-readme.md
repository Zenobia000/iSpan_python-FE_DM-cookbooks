# AI 協作決策分析技能

支援 Claude Code、Codex 與 OpenCode。安裝一次後，在自己開的資料夾研究一個分析題目：初始化、定義問題、人類確認、研究計畫、分析與交付。

本分支只提供可安裝技能，開發來源與完整文件見 [開發分支](https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/tree/feat/agentic-decision-analysis)。

## 安裝

使用 Node.js 22.20 以上與 skills CLI。下列命令針對輕量 `skills` 發行分支；選擇你使用的 CLI，勾選全部九個技能：

```bash
npx skills add https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/tree/skills -g
```

也可指定其中一個 CLI，非互動安裝全部技能：

```bash
# Codex
npx skills add https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/tree/skills --skill '*' -a codex -g -y

# Claude Code
npx skills add https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/tree/skills --skill '*' -a claude-code -g -y

# OpenCode
npx skills add https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/tree/skills --skill '*' -a opencode -g -y
```

安裝器處理各 CLI 的技能位置，技能不依賴固定家目錄。若新入口尚未顯示，重開 CLI 對話。安裝工具依據：[skills CLI](https://github.com/vercel-labs/skills)。

## 開始分析

自己建立或開啟題目資料夾，再在 CLI 對話中輸入：

| CLI | 第一次初始化 | 定義研究問題 |
| --- | --- | --- |
| Codex | `$ds-init 初始化目前專案` | `$ds-frame 我想研究……，資料在……` |
| Claude Code | `/ds-init 初始化目前專案` | `/ds-frame 我想研究……，資料在……` |
| OpenCode | `使用 ds-init skill 初始化目前專案，並建立 OpenCode 指令` | `/ds-frame 我想研究……，資料在……` |

OpenCode 的 skill 與 slash command 分開。首次用自然語言要求載入 ds-init；AI 應執行內附腳本並加上 `--opencode-commands`。建立後可用 `/ds-init`、`/ds-frame`、`/ds-eda` 等指令，也可直接要求原生 skill 工具載入角色。依據：[OpenCode skills](https://opencode.ai/docs/skills/)、[commands](https://opencode.ai/docs/commands/)。

初始化只需 Python 3.9 以上，建立：

```text
目前分析專案/
├── AGENTS.md、CLAUDE.md、.gitignore
├── research/          問題、計畫、假設、發現、來源與交接
├── data/              raw／interim／processed
├── analysis/          Python、SQL、notebook
├── config/            分組、期間、切分與成本參數
└── reports/           每輪結果、圖表與報告
```

重跑只補缺少的文件；既有文件與指令保留。路徑衝突會回報，不強制覆寫。初始化不下載資料或安裝分析套件；分析時沿用題目環境，缺少必要依賴時再明確處理。

AI 先整理並推薦完整研究問題，取得你的確認後才擬定研究計畫。計畫包含假設來源、替代解釋、反證、推薦順序及停止條件。已確認範圍內可連續分析，並自然追問。

## 九個技能

| 技能 | 用途 |
| --- | --- |
| ds-init | 建立單一分析專案骨架 |
| decision-analysis | 推薦下一個角色與理由 |
| ds-frame | 定義問題、取得確認、擬定研究計畫 |
| ds-proxy | 推演有理由且可驗證的代理特徵 |
| ds-eda | 探索資料、比較群組、檢查假設 |
| ds-rules | 提取並驗證可解釋候選規則 |
| ds-impact | 比較成本、效益情境與 ROI |
| ds-review | 核對實際資料、計算與報告 |
| ds-deliver | 整理決策報告、限制與交接 |

references 與模板已隨技能封裝，不需下載教材、PDF 或其他專案。三個 CLI 共用問題、計畫與交接文件；換工具時先讀 research/question.md、plan.md、handoff.md。

資料、程式、設定與報告屬於你的專案。技能更新不改寫既有分析成果。原始資料與產出預設忽略，分享前仍需逐項檢查研究文件、程式與設定。

此套件提供方法與檔案操作，未替代資料品質、統計設計及實際證據審查。角色切換不代表獨立審查。
