# Hypothread｜問脈 技能使用指南

目前資料夾就是一個分析專案。資料、程式與報告直接放在這裡；新題目由你另外開新資料夾。安裝技能後用 ds-init 建立骨架，再直接描述研究需求。詳見 [安裝指南](../../docs/distribution-readme.md)。

Claude Code 使用 `/ds-frame` 等入口；OpenCode 首次用自然語言叫用 ds-init 建立指令，之後可用 `/ds-frame` 或原生 skill 載入。若工具未顯示技能，核對安裝位置或重開對話。

## 如何開始

先在你的題目資料夾初始化：Codex 使用 `$ds-init`，Claude Code 使用 `/ds-init`，OpenCode 說「使用 ds-init skill 初始化目前專案並建立 OpenCode 指令」。重跑會保留既有檔案。

```text
使用 $decision-analysis。
我要研究夜班良率偏低的問題，資料在 data/raw/quality.csv。
請先釐清決策需求、研究問題與資料限制，和我確認後再擬定分析計畫。
```

這是提問示例，專案未附上該資料。換成你實際的問題與路徑即可。

AI 先整理問題、範圍與關鍵未知，取得你的確認，再提出有依據的分析方向。你提供領域校正與決策邊界；AI 負責研究推論、方法選擇與執行。已確認的範圍內可以連續工作，每張圖不必重新批准。

## 技能總覽

| 技能 | 適用情境 | 主要產出 |
| --- | --- | --- |
| ds-init | 新資料夾或要補齊骨架 | 研究、資料、程式、設定與報告目錄 |
| decision-analysis | 不確定從何開始，或接續既有研究 | 推薦下一個角色與理由 |
| ds-frame | 新題目、目標模糊或研究範圍改變 | 問題確認、研究計畫與假設 |
| ds-proxy | 缺少直接量測，需尋找代理訊號 | 候選代理與驗證設計 |
| ds-eda | 探索資料、比較群組、檢查假設 | 各輪分析、圖表與觀測證據 |
| ds-rules | 要把分析轉成候選警報或 SOP | 可解釋條件與驗證結果 |
| ds-impact | 要比較行動成本或估計效益 | 情境、ROI 與敏感度 |
| ds-review | 要核對分析或交付前找問題 | 審查意見及待修正事項 |
| ds-deliver | 要向決策者交付成果 | 報告與交接文件 |

不用依序跑完所有技能。只做探索也可以交付，不強制建模型或估 ROI。

## decision-analysis：幫你選擇下一步

```text
使用 $decision-analysis。先讀本專案的 research/ 與目前報告，
說明目前能回答什麼、最重要的證據缺口，以及推薦先處理哪件事。
```

新題目通常先進入 ds-frame；既有分析則沿用可核對的證據。它是角色路由，不會建立另一個專案或啟動固定管線。

## ds-frame：先想清楚問題

```text
使用 $ds-frame。我想知道夜班良率較低是否值得安排製程調查。
請先整理研究問題、比較範圍及關鍵假設，和我確認後再擬定計畫。
```

AI 先盤點資料與可行性，將待確認問題寫入 `research/question.md`。取得實際回覆後記錄確認來源與日期，再寫 `research/plan.md`、`hypotheses.md` 與需要的 `sources.md`。

計畫須從已知事實推導假設，交代替代解釋、反證條件、方法、資料切分與停止條件。例如先核對產品組成，再做可比條件下的班別比較，說明各步能排除什麼解釋。

方法依據見 [研究設計](../../skills/ds-frame/references/research-design.md)，其中區分彭明輝公開來源與本專案的操作化設計。

## ds-proxy：缺資料時提出可驗證替代方案

```text
使用 $ds-proxy。目前沒有震動訊號，只有電流、溫度與負載。
請評估哪些代理具有機制理由、在決策當下可取得，並設計驗證方式。
```

產出 `research/proxy.md`，逐項列出公式、單位、來源、替代解釋及失效條件。代理仍是假設，不能宣稱已量到缺少的物理量。

## ds-eda：帶著問題探索

```text
使用 $ds-eda。依已確認的研究問題與計畫，
先檢查產品組成能否解釋班別差異，再比較可比群組。
請回報效果量、樣本數、不確定性及仍未排除的解釋。
```

每輪結果放 `reports/explorations/round-001/` 等目錄；程式放 `analysis/`，設定放 `config/`，觀測結果彙整到 `research/findings.md`。AI 依資料結構讀取方法 references，不能只依 p-value 選圖或宣稱因果。

你可以自然追問：

```text
差異是不是集中在少數設備？請先判斷這個解釋是否值得查，
說明推薦的比較及理由，再在原計畫範圍內接續。
```

推薦下一步由 AI 說明理由；改變研究目標或資料使用範圍時，才重新確認問題。

## ds-rules：提取候選警報

```text
使用 $ds-rules。根據已驗證的訊號建立容易理解的候選條件，
保留後期資料驗證，回報涵蓋率、誤報、漏報與可行處置。
```

產出 `reports/rules.md`。列出公式、單位、決策當下可取得性及驗證設計。候選規則尚需領域確認與試行，不直接成為正式 SOP。

## ds-impact：把證據連到行動成本

```text
使用 $ds-impact。請先區分現有成本資料與待驗證假設，
比較維持現狀及試行警報的保守、基準、樂觀情境。
```

產出 `reports/impact.md`，列出共同期間、公式、成本與效果來源、敏感度及損益兩平。缺成本就說明缺口；預估節省與實測收益分列。

## ds-review：核對實際證據

```text
使用 $ds-review。請核對 reports/decision.md 與實際資料、程式及結果，
找出會改變建議的問題，將審查結論與修正要求寫入 reports/review.md。
```

檢查洩漏、方法假設、不確定性、多重比較、圖表、規則與成本口徑。審查記錄要指出對應報告、日期、證據位置與未解事項；修改後重查受影響的部分。自我審查會明示，換角色不等於獨立審查。

## ds-deliver：整理可讀的交付

```text
使用 $ds-deliver。讀者是現場主管，請先回答研究問題，
整理建議、證據、限制與下一步，完成審查並更新交接。
```

依內容選探索摘要或決策報告模板，寫入 `reports/decision.md`，搭配 `reports/review.md` 與 `research/handoff.md`。沒有做過的驗證或介入不填成已完成。

## 資料、文件與接續

```text
目前專案/
├── research/     question、plan、hypotheses、findings、sources、handoff
├── data/         raw／interim／processed
├── analysis/     Python、SQL、notebook
├── config/       期間、分組、資料切分與成本參數
└── reports/      explorations／figures／tables、decision.md、review.md
```

換對話或工具後直接說：

```text
請先讀 research/question.md、research/plan.md、research/handoff.md，
核對相關證據後接續推薦的下一步。
```

資料、程式與報告互相引用，以路徑、日期和版本識別來源。保留失敗假設與各輪結果，避免反覆重做或將事後發現改寫成事前假設。

## 為什麼有 skills/ 和 .agents/skills/

`skills/` 是唯一維護來源，包含角色模板、方法 references 與報告骨架；這裡供 Codex／OpenCode 共用，`.claude/skills/` 供 Claude Code 載入；所需方法資源已封裝在各技能內。`.opencode/commands/` 只負責載入對應 skill。修改角色後由維護者執行 `python3 tools/render_skills.py --write` 同步。

報告骨架固定必要的證據欄位，敘事依問題調整。圖表預設使用 Seaborn／Matplotlib，遵守 [圖表契約](../../skills/_shared/references/chart-contract.md)；不是每次固定產出相同圖型。

完整資源見 [reference 與模板索引](../../docs/skill-resources.md)，方法流程見 [研究流程](../../docs/research-workflow.md)。
