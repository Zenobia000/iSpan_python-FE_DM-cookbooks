# 分析技能操作流程

目前資料夾就是一個分析題目。直接在 Claude Code／Codex 對話中使用技能，將研究、資料、程式與報告保存在眼前這份專案。

## 技能入口

| 工作 | Claude Code | Codex |
| --- | --- | --- |
| 選擇下一個角色與續作 | `/decision-analysis` | `$decision-analysis` |
| 定義問題與研究計畫 | `/ds-frame` | `$ds-frame` |
| 代理特徵假設 | `/ds-proxy` | `$ds-proxy` |
| EDA 與假設檢查 | `/ds-eda` | `$ds-eda` |
| 候選規則 | `/ds-rules` | `$ds-rules` |
| 成本與效益情境 | `/ds-impact` | `$ds-impact` |
| 審查分析與報告 | `/ds-review` | `$ds-review` |
| 整理交付 | `/ds-deliver` | `$ds-deliver` |

例如：「使用 `$ds-frame`，資料在 data/raw/，先推薦研究問題並和我確認，之後擬定有假設與反證的分析計畫。」

若當次 host 未發現技能，可重開對話，或請 agent 讀取對應 skills/<名稱>/SKILL.md.tmpl 與 skills/_shared/session.md，依文件續做並說明載入方式。

## 一次分析如何進行

1. AI 讀需求、Schema 與已有研究文件，將問題寫入 research/question.md，說明推薦定義及理由。
2. 人類確認或修正問題；AI 保存實際回覆與確認的問題內容，不代為同意。
3. AI 擬定 research/plan.md，記下來源、假設、替代解釋、方法、順序與停止條件。
4. 在確認範圍內執行分析，每輪結果放 reports/explorations/<輪次>/，程式與參數放 analysis/、config/。
5. 結果與假設狀態寫入 research/findings.md，保留反證與失敗假設，不回填原研究假說。
6. 收斂後整理 reports/decision.md，由 ds-review 核對報告及實際證據，審查記在 reports/review.md。
7. 更新 research/handoff.md，交付報告、證據路徑與未解事項。

只做 EDA 也能交付，不必補出規則或 ROI。計畫範圍內可以連續工作，核心問題、資料使用或業務行動有重大改變時，才提出修訂並重新確認。

## 換對話怎麼接續

```text
使用 $decision-analysis 接續目前專案。
先讀 research/question.md、research/plan.md 與 research/handoff.md，
核對現有資料與結果，從未完成的地方繼續。
```

辨識版本可使用來源日期、檔名、Git commit 或需要的雜湊。資料、方法或結論改變時，重查受影響的比較與報告；不要只因舊文件寫過「通過」就沿用結論。

## 方法與產出依據

角色需要交付的內容見 [證據契約](harness-contracts.md)，按需讀取的 references 與報告骨架見 [技能參考與模板](skill-resources.md)。

技能與文件提供工作約定，品質依實際計算及審查判斷。自我審查要標明；不同角色名稱不代表不同模型或獨立驗證。檔案位置也不等於作業系統沙盒，權限依實際執行環境。

## 給技能維護者

技能只維護在 skills/，透過 tools/render_skills.py 產生 .agents/ 與 .claude/ 入口；使用分析技能不需要先執行這個產生工具。

```bash
python3 tools/render_skills.py --check
python3 -m unittest discover -s tests -v
```

上述檢查驗證入口與產生工具；真實分析行為仍依 [評估情境](../evals/scenarios.md)另外驗證。
