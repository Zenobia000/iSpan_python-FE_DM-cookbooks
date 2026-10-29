# 🎨 概念示意圖提示詞 — 大檔案分塊處理

> 對應 notebook：`01_chunking_large_files.ipynb`（模組 M02 · 資料清理）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：記憶體瓶頸 / 分塊流程 / 統計匯總
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_02_data_cleaning/notebooks
```

---

### 圖 1 · 記憶體瓶頸：一次讀 vs 分塊讀
目的：直觀展示大檔案為何無法一次讀入，以及分塊的優勢。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分為左右對比。左側標題『一次讀入（失敗）』，顯示一個巨大的資料檔案檔案圖示，箭頭指向一個電腦記憶體 RAM 圖示，記憶體爆炸 X 標記，標註繁體中文『MemoryError』『RAM 超載』。右側標題『分塊讀取（成功）』，顯示相同的檔案，被分割成多個小塊，逐次進入記憶體，標註繁體中文『chunk 1』『chunk 2』『chunk 3』，最後輸出結果。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 01_chunking_large_files_fig1_memory --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · Chunksize 迭代流程
目的：串起「開啟檔案 → 逐塊讀取 → 迴圈處理」的標準模式。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由上至下的流程圖。頂部為 CSV 檔案圖示，下方標註繁體中文『read_csv(chunksize=N)』；箭頭指向一個迭代器圖示；再箭頭指向三個並排的小資料塊，各標註『Chunk 1』『Chunk 2』『Chunk 3』；底部顯示一個 for 迴圈框，內含『處理各塊』的流程，最後匯總結果。柔和粉彩配色、白色背景、乾淨的流程圖風格、繁體中文標籤清晰。" --name 01_chunking_large_files_fig2_iteration --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 分塊統計與匯總
目的：展示「逐塊計算 → 累積 → 最終合併」的聚合邏輯。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，顯示三個資料塊分別進行統計計算（例如計算和、計數）。頂部分別標註『Chunk 1：年齡和=1000，人數=50』『Chunk 2：年齡和=1100，人數=45』『Chunk 3：年齡和=900，人數=55』；箭頭指向中央一個計算框，標註繁體中文『累積計算』『總年齡和』『總人數』；底部顯示結果『平均年齡=20.5』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 01_chunking_large_files_fig3_aggregation --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
