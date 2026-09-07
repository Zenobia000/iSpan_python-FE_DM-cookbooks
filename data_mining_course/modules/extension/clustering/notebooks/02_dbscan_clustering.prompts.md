# 🎨 概念示意圖提示詞 — DBSCAN 聚類

> 對應 notebook：`02_dbscan_clustering.ipynb`（模組 M10 · 資料探勘應用）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：密度概念與點分類 / DBSCAN 核心參數 / K-Means vs DBSCAN
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_10_data_mining_applications/notebooks/02_clustering
```

---

### 圖 1 · 密度概念：核心點、邊界點、噪音點

目的：展示 DBSCAN 如何根據點的局部密度將其分類為三種類型，視覺化密度聚類的核心思想。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示一個二維散點圖。密集區域的點標記『核心點』（紅），周圍的『邊界點』（黃），孤立的『噪音點』（灰）。用虛線圓圈表示範圍。柔和粉彩配色、白色背景、資訊圖表(infographic)風格。" --name 02_dbscan_clustering_fig1_points --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · DBSCAN 核心參數：eps 與 min_samples

目的：展示 eps（半徑）和 min_samples（鄰域最小點數）兩個關鍵參數如何控制簇的發現。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示三個並排的二維散點圖場景。左『小eps』；中『適當eps』；右『大eps』。每圖內核心點周邊圓圈表示搜索範圍。簡潔標籤。柔和粉彩配色、白色背景、infographic 風格。" --name 02_dbscan_clustering_fig2_params --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · K-Means vs DBSCAN 優缺點對比

目的：比較兩種聚類方法在簇形狀、噪音處理、參數調整方面的特點。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分為左右兩欄對比。左『K-Means』：球形簇、固定K；右『DBSCAN』：任意形狀、自動尋找。下方列出『優點』與『缺點』。柔和粉彩配色、白色背景、infographic 風格。" --name 02_dbscan_clustering_fig3_comparison --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
