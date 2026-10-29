# 🎨 概念示意圖提示詞 — K-Means 聚類

> 對應 notebook：`01_kmeans_clustering.ipynb`（模組 M10 · 資料探勘應用）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：K-Means 核心概念 / 演算法迭代過程 / 肘部法則選 K
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_10_data_mining_applications/notebooks/02_clustering
```

---

### 圖 1 · K-Means 核心概念：質心與簇

目的：視覺化 K-Means 將數據點分配到 K 個簇、每個簇有一個質心的基本思想。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示一個二維散點圖，用三種顏色分別圍繞三個簇。每個簇中心標註『質心』。標註繁體中文『點』『質心』『簇』。柔和粉彩配色、白色背景、資訊圖表(infographic)風格。" --name 01_kmeans_clustering_fig1_concept --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · K-Means 迭代演算過程

目的：展示初始化、分配、更新質心的三步迭代流程，直到收斂。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，四個並排的步驟卡片。『初始化』『分配』『更新』『完成』。各步驟用箭頭連接。簡潔標籤。柔和粉彩配色、白色背景、infographic 風格。" --name 01_kmeans_clustering_fig2_iteration --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 肘部法則：選擇最佳 K 值

目的：展示用肘部法則判斷最佳簇數量的圖表特徵，幫助理解何時停止增加 K。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，一張折線圖展示 K 值（X軸）vs 誤差（Y軸）的關係。曲線遞減，在某個位置有明顯『肘部』。標註『肘部』與『最佳K』。柔和粉彩配色、白色背景、資訊圖表風格。" --name 01_kmeans_clustering_fig3_elbow --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
