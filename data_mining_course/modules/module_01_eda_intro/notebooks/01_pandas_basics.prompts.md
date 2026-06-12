# 🎨 概念示意圖提示詞 — Pandas 基礎操作複習

> 對應 notebook：`01_pandas_basics.ipynb`（模組 M01 · 探索性資料分析入門）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：核心結構對照 / 載入檢視流程 / 索引選取
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_01_eda_intro/notebooks
```

---

### 圖 1 · 核心結構對照：Series vs DataFrame
目的：一眼看懂 Pandas 兩大資料結構的維度差異。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，並排對比 Pandas 兩種核心資料結構。左側為單欄帶索引的直立清單，代表一維的 Series，標註繁體中文『Series（一維）』與『索引 index』；右側為三欄四列的彩色表格，代表二維的 DataFrame，標註繁體中文『DataFrame（二維表格）』『欄 columns』『列 rows』。柔和粉彩配色、白色背景、乾淨的資訊圖表(infographic)風格、繁體中文標籤清晰、無多餘裝飾。" --name 01_pandas_basics_fig1_structures --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 資料載入與檢視流程
目的：串起「讀檔 → DataFrame → 快速檢視」的標準起手式。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右的水平流程圖。第一格為 CSV 檔案圖示，箭頭上方標註 read_csv；第二格為一個彩色 DataFrame 表格；再接一個箭頭；第三格為三張並排小卡，分別標註繁體中文『head() 看前幾列』『info() 看欄位型別』『describe() 看統計摘要』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 01_pandas_basics_fig2_load_inspect --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 資料選取：loc 與 iloc
目的：區分「依標籤」與「依位置」兩種選取邏輯。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，一個五列四欄的資料表格，左側標示列索引、上方標示欄名。用藍色外框框選某一整列並標註繁體中文『loc：依標籤選取』；用橘色外框框選某一整欄並標註繁體中文『iloc：依整數位置選取』。柔和粉彩配色、白色背景、乾淨教學圖表、繁體中文標籤清晰。" --name 01_pandas_basics_fig3_loc_iloc --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
