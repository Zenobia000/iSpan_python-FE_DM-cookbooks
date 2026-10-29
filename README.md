# AI 協作決策分析

一個分析題目就是一個專案，直接在目前資料夾透過 Claude Code／Codex 與 AI 工作。AI 先釐清問題並和人類確認，再推導研究計畫、執行分析、整理證據與決策報告。新題目由你另開新資料夾。

## 怎麼開始

在目前專案開啟 Codex，直接輸入：

```text
使用 $decision-analysis。
我要研究 <問題描述>，資料在 <資料路徑>，欄位定義在 <Schema 路徑>。
先幫我想清楚並推薦研究問題，和我確認後再擬定分析計畫。
```

Claude Code 使用 `/decision-analysis`。你也可以直接指定 `$ds-frame`、`$ds-eda` 等角色；技能會讀取目前專案的研究文件接續。

```text
定義問題 → 人類確認 → AI 擬定研究計畫
                         ↓
                   分析與反證檢查
                         ↓
                 報告 → 審查 → 交付
```

分析可以反覆迭代，不強制建模型、提規則或算 ROI。

## 文件放在哪裡

```text
目前專案/
├── research/          問題、計畫、假設、來源與交接
├── data/              raw／interim／processed
├── analysis/          Python、SQL、notebook
├── config/            分組、期間、切分與成本參數
├── reports/           每輪結果、圖表、決策與審查報告
├── skills/            技能來源、方法 references 與報告模板
├── .agents/skills/    Codex 技能入口
├── .claude/skills/    Claude Code 技能入口
├── docs/              使用說明、方法與白皮書
├── tools/             技能入口產生工具，供維護者使用
├── tests/             技能產生工具測試
└── archive/           歷史教材與素材
```

`research/question.md` 記問題與實際確認，`research/plan.md` 記研究方向；假設定義與觀測後結果分別放 hypotheses.md、findings.md。換對話時讀 `research/handoff.md` 接續。這些文件目前是空白骨架，開始分析時由 AI 依實際題目填寫。

原始資料與報告產出預設不提交；分享前仍須逐項檢查研究文件、程式及參數是否包含敏感資訊。

## 想了解更多

| 想知道什麼 | 文件 |
| --- | --- |
| 每個技能什麼時候用、怎麼提問？ | [技能使用指南](.agents/skills/README.md) |
| 從問題確認到研究與交付怎麼做？ | [研究流程](docs/research-workflow.md) |
| 分析時如何操作與接續？ | [技能操作](docs/harness.md) |
| References 與報告骨架何時讀？ | [技能參考與模板](docs/skill-resources.md) |
| 為什麼採用這個架構？ | [設計取捨](docs/harness-design.md) |
| 下一版如何安裝技能並初始化專案？ | [封裝設計規格](docs/skill-distribution-design.md)（尚未實作） |
| 全部文件在哪裡？ | [文件地圖](docs/README.md) |

## 維護技能

平常使用者直接對話即可。維護技能內容時修改 `skills/<角色>/SKILL.md.tmpl`，再產生兩個 host 的入口：

```bash
python3 tools/render_skills.py --write
python3 tools/render_skills.py --check
python3 -m unittest discover -s tests -v
```

需要資料分析套件時依根目錄 pyproject.toml 與 uv.lock 執行 `uv sync --locked`。歷史教材見 [archive/](archive/README.md)，不作為目前分析的預設流程。
