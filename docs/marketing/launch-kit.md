# Hypothread｜問脈 推廣素材包

## 對外入口

- 產品與安裝：[Hypothread 發行分支](https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/tree/skills)
- 英文介紹：[English README](https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/blob/skills/README.en.md)
- 主視覺：[hero.png](../../assets/brand/hero.png)
- Icon：[icon.png](../../assets/brand/icon.png)
- 問題與使用回饋：[GitHub Issues](https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/issues)

下面文案已備妥，適合人工貼上；本次未代為發布社群貼文、寄信或聯絡第三方。GitHub 的 README、About、topics 與 v0.2.1 release 作為本次公開入口。

## 一句、短版與完整介紹

一句介紹：

> Hypothread｜問脈：給 Claude Code、Codex、OpenCode 的資料科學技能，讓 AI 先想清楚問題，再沿著證據持續探勘。

短版介紹：

> 分析資料時，你會一路追問產品組成、設備差異與時間效應。Hypothread 把這些追問接到研究問題、假設、反證與實際計算，將每輪發現保存在自己的專案裡。安裝九個技能，從初始化、問題確認、探索分析到報告交接，按題目需要使用。

完整介紹：

> Hypothread｜問脈是一套以研究方法為核心的資料科學 AI 協作技能，支援 Claude Code、Codex 與 OpenCode。
>
> AI 先整理決策需求、已知與未知，推薦完整研究問題並取得人類確認，再推導有理由的假設、替代解釋與分析順序。你提供領域判斷，AI 執行計算與繪圖；每次追問都能接著已有證據往下走。
>
> 套件提供九個技能、方法 references、報告模板與單一題目初始化。資料、程式、參數與成果留在你的專案裡，換對話或工具時也能讀取交接繼續。只做 EDA 也可以收斂交付。

## 中文首發貼文

建議搭配 hero.png，連結指向 skills 發行分支。

```text
分析資料時，我最想保留的是追問的能力。

夜班良率比較低，是產品不同、設備不同，還是時間效應？
有了 AI，寫程式更容易；我希望研究問題、假設和反證，也能被好好接起來。

所以做了 Hypothread｜問脈。

它是給 Claude Code、Codex、OpenCode 的九個資料科學技能：
先定義問題、和人確認，再由 AI 推導研究計畫並持續探勘。
方法、計算、失敗假設與報告都保留在自己的分析專案裡。

你不用先學一套新的多專案管理介面。
開一個題目資料夾，安裝技能，從 ds-init、ds-frame 開始。

目前已驗證安裝、初始化與檔案流程；模型分析行為還需要更多實際題目回饋。

你最想交給 AI 一起追查的資料問題是什麼？
https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/tree/skills
```

## 國際社群短版

```text
I built Hypothread: research-first data science skills for Claude Code, Codex, and OpenCode.

Frame the question, confirm the scope, develop hypotheses and alternative explanations, then keep exploring with an evidence trail in your own project.

Nine skills cover project initialization, research design, EDA, interpretable rules, impact scenarios, review, and handoff. Method references and report templates ship with the skills.

The bundled research instructions are primarily in Traditional Chinese. Installation and workspace behavior have been tested; full model-driven analysis validation is still ongoing.

Try it with one question you want to investigate:
https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks/tree/skills
```

## 60 秒展示腳本

| 時間 | 畫面與內容 | 說明 |
| --- | --- | --- |
| 0–10 秒 | 展示真實題目的提問與資料欄位 | 使用合成或可公開資料，說明要支援的決定 |
| 10–20 秒 | 載入 ds-init | 建立目前專案，保留原有檔案 |
| 20–35 秒 | 載入 ds-frame，AI 整理問題 | 重點放在人類確認前的研究推論 |
| 35–50 秒 | 顯示 plan、hypotheses 及一輪探索 | 只展示實際執行過的輸出，指出替代解釋 |
| 50–60 秒 | 顯示 handoff 與安裝指令 | 邀請以自己的題目試用，回報卡住的位置 |

這是待錄製的腳本，不是已存在的影片。尚未錄製時用 README 的示意對話介紹，不把它標成實測 demo。

## 發布後一週的具體安排

| 時點 | 動作 | 要觀察什麼 |
| --- | --- | --- |
| 首日 | GitHub 品牌 README、英文版、公開 release、About 與 topics | 使用者是否能找到產品入口與安裝指令 |
| 第 2 日 | 以主視覺搭配中文首發文，分享給既有資料科學讀者 | 訪客是否能說出套件用途，而非只記得圖像 |
| 第 3–4 日 | 發布一個可重現的公開資料研究案例 | 問題確認、追問與證據交接是否實際有用 |
| 第 5 日 | 將案例與英文版分享到合適的開源／分析社群 | 收集不同 CLI 的安裝與第一輪任務回饋 |
| 第 7 日 | 整理回饋、修正文案或最常見阻礙 | 是否順利初始化、是否完成首次問題確認、哪一步放棄 |

這是人工執行計畫，沒有建立排程或自動發布。

## GitHub 搜尋與介紹

建議 About：

> Hypothread — research-first data science skills for Claude Code, Codex, and OpenCode. Frame questions, explore evidence, and hand off decisions.

建議 topics：data-science、agent-skills、claude-code、codex、opencode、exploratory-data-analysis、feature-engineering、decision-analysis、research-methods。

現有 repository 名稱仍包含原課程名稱，對外貼文統一指向 /tree/skills。若日後決定另建 Hypothread 專屬 repository，再做來源與連結遷移；本次不改名或更換預設分支。

## 回饋與衡量

先問試用者使用的 CLI、想研究的問題、是否完成初始化與問題確認，以及卡住的具體位置。不要求提供商業原始資料或憑證。

GitHub 的可見資料可用來觀察頁面造訪、clone 與 stars；clone 不等於完成安裝，star 不等於實際使用。首次分析完成率需要自願回饋或另外設計追蹤，本套件未加入分析資料上傳或使用者追蹤。
