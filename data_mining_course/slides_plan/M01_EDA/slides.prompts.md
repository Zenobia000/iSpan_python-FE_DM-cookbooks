# 概念示意圖提示詞：第 1 章　探索式資料分析

> 對應講義：[`M1_Fundamentals_of_Systematic_EDA.md`](../../modules/module_01_eda_intro/slides/M1_Fundamentals_of_Systematic_EDA.md)
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 2 張：EDA 六步檢查表 / 選圖決策樹
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；標籤要完全清晰再改 `--quality high`。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/slides_plan/M01_EDA
```

---

### 圖 1 · EDA 六步檢查表
用在第 1 張。目的：把系統化掃描壓成一張可以照著做的檢查表。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，一張由上而下的六格檢查清單，每格左邊一個核取方塊。六格依序為『讀檔與看形狀』『欄位型態與缺失值比例』『單變量分佈』『兩變量關係』『異常與重複』『寫下要回答的問題』。最後一格用橘色強調並標註『這才是產出』。右側放一個放大鏡與表格的小圖示。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M1_fig1_eda_checklist --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 選圖決策樹
用在第 3 張。目的：用型別與問題型態決定圖表，取代把圖表一種一種介紹。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，一個由左至右展開的決策樹。根節點寫『你要回答什麼』，分成三個分支。分支一『看一個變數的分佈』指向直方圖與盒鬚圖的小圖示。分支二『看兩個變數的關係』再分成『數值對數值』指向散佈圖、『類別對數值』指向分組盒鬚圖。分支三『看隨時間變化』指向折線圖。每個葉節點下方一行小字寫用途。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M1_fig2_chart_decision --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
