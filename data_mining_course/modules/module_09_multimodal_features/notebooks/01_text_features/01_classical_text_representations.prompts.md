# 🎨 概念示意圖提示詞 — 經典文本表示法快速回顧

> 對應 notebook：`01_classical_text_representations.ipynb`（模組 M09 · 多模態特徵（文字））
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：經典表示法對比框架 / 詞袋與 TF-IDF 差異 / 靜態詞向量一詞多義問題
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/01_text_features
```

---

### 圖 1 · 三大經典文本表示法全景對比

目的：一眼看懂詞袋 (BoW)、TF-IDF、靜態詞向量的輸出形態與核心差異。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，並排展示三種經典文本表示法。左側為稀疏矩陣，標示繁體中文『詞袋』，顯示文件-詞項計數；中間為加權矩陣，標示『TF-IDF』，數值為權重分數；右側為三個圓形向量，標示『詞向量』，代表稠密向量。柔和粉彩配色(orange/blue/green)、白色背景、乾淨的資訊圖表風格、繁體中文標籤清晰、無多餘裝飾。" --name 01_classical_text_representations_fig1_overview --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 詞袋 (BoW) vs TF-IDF：如何給詞加權

目的：展示 BoW 的「機械計數」與 TF-IDF 的「智慧加權」差異。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分左右兩格展示同一篇文件。左格標示『BoW：計數』，列出詞彙與數字，所有詞等同對待；右格標示『TF-IDF：加權』，同樣詞彙但虛詞被淡化，實詞被加強，用顏色深淺表示權重。中間箭頭標示『加權』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 01_classical_text_representations_fig2_bow_tfidf --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 靜態詞向量的一詞多義困境

目的：直觀呈現靜態詞向量無法區分 "bank" 的兩種意思的根本限制。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，上方展示兩個情境。左側為銀行室內，標示『銀行』；右側為河岸，標示『河岸』。下方顯示一個高維向量，中心標示『Word2Vec』，用紅色警示號『❌ 同一向量』。用箭頭表示兩個情境都指向同一個向量。柔和粉彩配色(藍/金)、白色背景、infographic 風格、繁體中文標籤清晰。" --name 01_classical_text_representations_fig3_polysemy --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
