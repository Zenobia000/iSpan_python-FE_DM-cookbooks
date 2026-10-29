# 技能參考文件與產出模板

研究問題確認與 plan 決定要研究什麼；references 協助 agent 選擇與檢查方法，templates 則讓每輪分析和交付保留必要證據。三者共同支援有依據的探索，不把每次追問變成讓人選方法的選單。

## 現有資源

```text
skills/
├── _shared/
│   ├── session.md
│   └── references/
│       ├── evidence-levels.md
│       ├── exploration-protocol.md
│       └── chart-contract.md
├── ds-frame/
│   ├── references/research-design.md
│   └── SKILL.md.tmpl                   問題確認與研究計畫
├── ds-eda/
│   ├── SKILL.md.tmpl
│   ├── references/
│   │   ├── data-understanding.md
│   │   ├── method-selection.md
│   │   ├── dependent-data.md
│   │   └── confounding-checks.md
│   └── templates/inquiry.md
└── ds-deliver/
    ├── SKILL.md.tmpl
    └── templates/
        ├── exploration-summary.md
        └── decision-report.md
```

其他角色入口仍保留原位置；此圖聚焦方法參考與模板。

## 何時讀什麼

| 工作情境 | 文件與用途 |
| --- | --- |
| 新題目或問題改變 | [research-design](../skills/ds-frame/references/research-design.md)：問題意識、來源比較、假設與反證；先確認問題再擬 plan |
| 判讀結果或寫建議 | [evidence-levels](../skills/_shared/references/evidence-levels.md)：區分假設、關聯、獨立驗證與介入成效 |
| 開始或接續一輪探勘 | [exploration-protocol](../skills/_shared/references/exploration-protocol.md)：問題、設計、結果、推薦下一步與交接 |
| 選圖、繪圖或審查圖表 | [chart-contract](../skills/_shared/references/chart-contract.md)：分母、尺度、單位、區間與可重現來源 |
| 新資料或品質疑慮 | [data-understanding](../skills/ds-eda/references/data-understanding.md)：觀測單位、時間可取得性、缺值、join 與覆蓋 |
| 為問題選方法 | [method-selection](../skills/ds-eda/references/method-selection.md)：估計目標、比較設計、推論條件與多重比較 |
| 多設備、重複量測或時間資料 | [dependent-data](../skills/ds-eda/references/dependent-data.md)：獨立單位、群集／時間結構與驗證 |
| 組成差異或替代解釋 | [confounding-checks](../skills/ds-eda/references/confounding-checks.md)：可比範圍、分層、敏感度及主張邊界 |

技能入口和共用 session 契約明確要求按需讀取 repository 根目錄的資源。這是 agent 執行規則，沒有新增會自動判讀 Schema 或檢查統計正確性的 runtime；必須由 agent 實際讀取、執行工具並留下證據。

## 模板用在哪裡

| 模板 | 使用情境 | 專案內產出 |
| --- | --- | --- |
| [inquiry](../skills/ds-eda/templates/inquiry.md) | 每輪有目的的比較或假設檢查 | `reports/explorations/round-001/eda.md` 等輪次文件 |
| [exploration-summary](../skills/ds-deliver/templates/exploration-summary.md) | EDA 或跨輪發現摘要，不需要強行加入 ROI | 正式草稿預設 `reports/decision.md`；另需摘要時可用 `reports/exploration-summary.md` |
| [decision-report](../skills/ds-deliver/templates/decision-report.md) | 候選行動、規則、效益或試行決策 | `reports/decision.md` |

研究骨架直接放在根目錄 `research/`。Agent 按需讀取上述模板，將實際內容寫入本專案的 `reports/`；不預填結果或建立另一層專案。

結構可依問題調整，證據與推論要求保持一致。沒有執行的項目說明範圍，不以空表、虛構數字或不存在的 review 填滿章節。正式交付不論使用哪個模板，都需對實際報告版本完成 review。

## 如何連成一次探索

```text
已確認問題與有效 plan
       ↓
依本輪資料與問題讀取必要 reference
       ↓
用 inquiry 記下設計 → 執行計算與圖表 → 填入結果
       ↓
更新 findings 與 handoff，推薦有理由的下一步
       ↓
收斂時選 exploration-summary 或 decision-report
       ↓
核對證據與圖表 → review → 相同版本交付
```

繪圖預設沿用 Seaborn／Matplotlib，必要時依依賴流程擴充工具；圖型由研究問題決定。原始統計表、圖表與文字使用一致估計，軸、分母、不確定性及來源都要可核對。

## 維護與驗證

資源只維護在 `skills/`，`.agents/` 與 `.claude/` 入口引用同一份來源。調整技能本文後用 `python3 tools/render_skills.py --write` 產生入口，再用 `--check` 核對漂移。

Markdown 存在、連結有效與 skill 格式通過，不能證明實際分析判斷正確。真實行為評估需檢查 agent 是否在適當情境讀取參考、選擇合理方法、保存不成立假設及維持主張邊界；情境見 [evals](../evals/scenarios.md)。
