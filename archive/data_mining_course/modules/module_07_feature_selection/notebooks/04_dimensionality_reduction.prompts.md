# 🎨 概念示意圖提示詞 — 降維 (Dimensionality Reduction)

> 對應 notebook：`04_dimensionality_reduction.ipynb`（模組 M07 · 特徵選擇）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：降維 vs 特徵選擇對比 / PCA 線性轉換與主成分 / t-SNE 非線性視覺化
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_07_feature_selection/notebooks
```

---

### 圖 1 · 降維 vs 特徵選擇：轉換 vs 挑選的本質區別

目的：清晰展示降維是特徵空間的線性/非線性轉換，而非選擇原始特徵的子集。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分為左右兩側對比。左側『特徵選擇 Feature Selection』顯示五個彩色圓圈代表五個原始特徵，箭頭指向三個圓圈被保留標註『保留原始特徵』『挑選子集』，底部標註『原始特徵空間 5 維 → 3 維』。右側『降維 Dimensionality Reduction』顯示五個彩色圓圈，經過扭轉變形的箭頭，轉換成三個新的組合圓圈，標註『組合融合』『新特徵空間』，底部標註『線性/非線性轉換』『5 維 → 3 維』。下方對比表格：『特徵選擇』欄標註『保留原始特徵、易解釋、忽視交互』；『降維』欄標註『合成新特徵、高度融合、可捕捉複雜結構』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 04_dimensionality_reduction_fig1_comparison --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · PCA：尋找最大變異方向的線性轉換

目的：展示 PCA 如何識別資料中變異最大的方向（主成分），並用低維坐標表示。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示二維到一維的 PCA 過程。左側『原始資料（2D）』顯示一個散點圖，約 15 個點分布在二維平面，呈橢圓形聚集。中央箭頭與數學符號『PC1, PC2』。右側『主成分空間（1D）』顯示同樣的點投影到一條斜的軸線上，標註『PC1 主成分 1』，該軸箭頭較長；垂直的『PC2 主成分 2』箭頭較短，標註『方差小，可舍棄』。底部數字標註『解釋變異度：PC1 = 85%, PC2 = 15%』。側邊視覺化顯示方差序列柱狀圖逐步遞減。標註繁體中文『線性正交轉換』『旋轉坐標軸』『保留最大變異』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 04_dimensionality_reduction_fig2_pca --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · t-SNE：非線性降維保留局部結構的視覺化

目的：展示 t-SNE 如何透過保留資料點之間的局部鄰近關係，將高維資料視覺化為二維或三維，特別適合發現聚類結構。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示高維空間到視覺化空間的 t-SNE 轉換。左側『高維資料（10D）』顯示一個抽象的高維點雲，點彩色各異，分為三個虛擬簇區域，但視覺上混亂無序。中央大箭頭標註『t-SNE 非線性轉換』『保留局部相似性』。右側『視覺化空間（2D）』顯示明顯的三個彩色聚類團，點聚集明顯，同色點靠近，不同顏色的點分開，底部標註『清晰聚類結構』『易於視覺化解釋』。上方對比框標註『PCA 線性，全局變異』『t-SNE 非線性，局部鄰域』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 04_dimensionality_reduction_fig3_tsne --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
