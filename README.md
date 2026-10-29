# AI 協作決策分析 Harness

在 Claude Code／Codex 中，以領域問題驅動資料分析：提出假設、執行計算、核對證據，再形成決策建議。本專案提供角色技能、本機狀態與版本綁定的審查交付；分析方法由 agent 依資料與問題選擇。

## 先看哪裡

| 你想做什麼 | 入口 |
| --- | --- |
| 不知道選哪個技能，想看情境與提示範例 | [技能使用指南](.agents/skills/README.md) |
| 新題目如何確認問題、分開資料與報告？ | [研究流程與專案目錄](docs/research-workflow.md) |
| 開始分析、接手已有任務 | [Harness 使用說明](docs/harness.md) |
| 理解整套架構與 gstack 設計取捨 | [架構設計](docs/harness-design.md) |
| 理解人機分工與分析方法 | [架構白皮書](docs/AI協作決策分析架構.md) |
| 找特徵工程、EDA 與選圖參考 | [方法參考](docs/references/README.md) |
| 尋找全部文件 | [文件地圖](docs/README.md) |
| 查看舊課程與歷史素材 | [封存索引](archive/README.md) |

## 目錄怎麼看

日常工作集中在六個目錄：`skills/` 定義方法責任、`tools/` 管理證據、`docs/` 解釋用法、`examples/` 提供計算範例、`tests/` 驗證程式、`evals/` 評估真實 agent 行為。

```text
.
├── README.md                 專案入口
├── AGENTS.md                 共用協作規範
├── CLAUDE.md                 Claude 指引入口
├── docs/                     使用、架構、契約與白皮書
│   ├── README.md             文件地圖
│   └── references/           特徵工程指南 PDF 與可編輯原稿
├── skills/                   技能單一維護來源
│   ├── _shared/              共用 session 契約
│   ├── decision-analysis/    任務路由
│   └── ds-*/                 定義、代理、EDA、規則、效益、審查、交付
├── tools/                    狀態、證據與技能產生工具
├── examples/                 選用的分析程式
│   └── factory_analysis/     單一機台計算範例
├── projects/                 各題獨立工作區，個別專案不提交
├── tests/                    Harness runtime 測試
├── evals/                    真實 agent 情境與評分規準
├── .agents/skills/           產生的 Codex 技能入口
├── .claude/skills/           產生的 Claude 技能入口
├── .github/workflows/        自動驗證
├── pyproject.toml            Python 環境與依賴
├── uv.lock                   鎖定版本
└── archive/                  舊教材、簡報與歷史素材
```

任務資料與執行紀錄放在本機 `projects/`，由 `.gitignore` 排除。技能內容改 `skills/`，再產生兩個 host 的入口。

## 最小使用流程

從 repository 根目錄啟動 Claude Code 或 Codex：

- Claude Code：`/decision-analysis`，或直接使用 `/ds-frame`、`/ds-eda` 等角色。
- Codex：使用 `$decision-analysis`，或指定 `$ds-frame`、`$ds-eda` 等角色。

例如：「我想找出哪些條件與良率下降有關。先讀 Schema，推薦研究問題並和我確認，再擬定有假設與反證的分析計畫。」

```text
業務問題 ↔ Claude Code／Codex
                    ↓ 選擇當前需要的角色
             問題定義 → 人類確認 → 研究計畫 → EDA
                         ├─ 可選：代理特徵／候選規則／效益情境
                         └─ 決策草稿 → review → 本機交付
                    ↕
             case 狀態、證據與版本
```

角色可反覆迭代；不要求每次都建模型、提規則或算 ROI。已存在的 case 先查狀態，只重做缺失或失效的部分。

Harness 工具僅需 Python 3.9 以上的標準函式庫：

```bash
python3 tools/ds_harness.py init --case projects/first --goal eda
python3 tools/ds_harness.py status --case projects/first
```

角色登錄、review 與交付命令見 [完整操作](docs/harness.md)。工具檢查版本和結構，統計方法與業務結論仍須審查。

## 開發與驗證

```bash
python3 tools/render_skills.py --check
python3 -m unittest discover -s tests -v
```

修改技能模板後，執行 `python3 tools/render_skills.py --write` 產生兩個 host 的入口。

只有使用計算範例時，才需要安裝分析環境：

```bash
uv sync --locked
uv run --locked python -m examples.factory_analysis demo --out /tmp/factory-demo-001
uv run --locked python -m unittest discover -s examples/factory_analysis/tests -v
```

依賴與 lockfile 沿用既有分析環境，其中保留封存教材所用套件。單一機台範例的資料限制見 [範例說明](examples/factory_analysis/README.md)。Runtime 測試與真實 host 行為評估分開回報，後者見 [evals/scenarios.md](evals/scenarios.md)。

## 舊內容去哪裡

原 `data_mining_course/`、舊課程首頁、行銷圖片、其他簡報及舊 Cursor 規約已集中至 [archive/](archive/README.md)。與現行架構直接相關的白皮書與特徵工程指南保留在 `docs/`，搬移對照見封存索引。
