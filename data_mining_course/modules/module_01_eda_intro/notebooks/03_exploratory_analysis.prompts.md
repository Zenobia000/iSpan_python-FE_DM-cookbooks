# 🎨 概念示意圖提示詞 — 系統化探索性資料分析（EDA）

> 對應 notebook：`03_exploratory_analysis.ipynb`（模組 M01 · 探索性資料分析入門）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：EDA 工作流程 / 資料品質掃描 / 單變數 vs 多變數分析
>
> ⚠️ 含中文字的圖預設 `--quality low`；標籤需完全清晰可改 `--quality high`。
> ▶️ 執行前先 `cd` 到本資料夾，圖輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_01_eda_intro/notebooks
```

---

### 圖 1 · EDA 工作流程五步驟
目的：把 EDA 視為一條有順序的系統化流程，而非隨意亂看。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右的五階段水平流程圖，每階段一個圓形圖示與繁體中文標籤：『1 載入資料』→『2 資料品質檢查』→『3 單變數分析』→『4 多變數分析』→『5 假設與洞見』，以箭頭串連。最上方總標題『系統化 EDA 工作流程』。柔和粉彩配色、白色背景、乾淨 infographic 風格、繁體中文標籤清晰。" --name 03_exploratory_analysis_fig1_workflow --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 資料品質掃描儀表板
目的：開工前先體檢資料的四大品質面向。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，一個簡潔的資料品質儀表板，四張並排檢查卡，各有小圖示與繁體中文標籤：『缺失值』『重複資料』『資料型別』『異常值』，每張卡上有勾選/警示的狀態標記。頂端標題『資料品質掃描』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 03_exploratory_analysis_fig2_quality_scan --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 單變數 vs 多變數分析
目的：分清「先看單一變數」與「再看變數間關係」兩個層次。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，畫面左右對照。左半標註繁體中文『單變數分析』，內含一個直方圖與一個箱型圖小圖示；右半標註繁體中文『多變數分析』，內含一個散點圖與一個相關係數熱圖小圖示；中間以箭頭由左指向右表示分析的推進。柔和粉彩配色、白色背景、乾淨教學圖表、繁體中文標籤清晰。" --name 03_exploratory_analysis_fig3_uni_vs_multi --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
