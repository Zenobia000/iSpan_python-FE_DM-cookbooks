# 🎨 概念示意圖提示詞 — 上下文嵌入與句向量

> 對應 notebook：`03_contextual_embeddings.ipynb`（模組 M09 · 多模態特徵（文字））
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：靜態 vs 上下文嵌入對比 / Token 嵌入與句向量池化 / 語意檢索應用
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/01_text_features
```

---

### 圖 1 · 靜態詞向量 vs 上下文嵌入的本質差異

目的：展示同一詞在不同句子中如何獲得不同向量表示。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，左側標示『Word2Vec』，兩個句子中的同一詞連到同一黃色向量，標註『同向量』。右側標示『BERT』，同詞在不同句子連到不同綠色向量，標註『上下文向量』。中間用對比箭頭分隔。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 03_contextual_embeddings_fig1_comparison --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 從 Token 嵌入到句向量：Mean Pooling

目的：視覺化 token 層級嵌入如何池化成單一句向量。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右展示轉換過程。左側為矩陣標示『Token 嵌入』，多個 token 各自的向量；中間標示『池化』，箭頭；右側為圓形向量標示『句向量』。柔和粉彩配色(藍→綠→橙)、白色背景、infographic 風格、繁體中文標籤清晰。" --name 03_contextual_embeddings_fig2_pooling --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 語意檢索應用：向量相似度匹配

目的：展示如何用句向量進行語意檢索，找出最相近的文件。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，左側為 Query 框標示『查詢』，編碼成圓形向量。右側為三篇文件各自為圓形向量。中間用顏色與分數連結，綠線『最相似』、藍線『相近』、灰線『不相似』，展示餘弦相似度排序。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 03_contextual_embeddings_fig3_semantic_search --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
