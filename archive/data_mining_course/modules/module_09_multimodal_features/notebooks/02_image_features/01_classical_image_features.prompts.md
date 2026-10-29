# 🎨 概念示意圖提示詞 — 經典影像特徵

> 對應 notebook：`01_classical_image_features.ipynb`（模組 M09 · 多模態特徵（圖像））
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：像素到向量的轉換 / 色彩直方圖 vs HOG 對比 / 手工特徵的限制
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/02_image_features
```

---

### 圖 1 · 像素到向量：影像的三維張量結構
目的：幫助讀者理解影像在電腦中的本質是 (H, W, C) 像素張量。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。左：JPG 檔案圖示（貓），中：箭頭，右：三維張量結構。張量標籤分別為『H 高度』『W 寬度』『C 通道』。用紅綠藍色塊表示 RGB 三層。白背景、粉彩配色。" --name 01_classical_image_features_fig1_tensor_structure --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 色彩直方圖 vs HOG 對比
目的：視覺化兩種經典手工特徵的不同提取方式與特性。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。左：彩色影像 + 三條直方圖（紅綠藍），標籤『色彩直方圖』。右：同影像 + 方格網與箭頭，標籤『HOG』。下方各有向量框：『256 維』『捕捉邊緣』。白背景、粉彩配色。" --name 01_classical_image_features_fig2_histograms_hog --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 手工特徵 vs 學習式特徵對比
目的：幫助讀者理解為什麼 2026 年已轉向學習式特徵（ViT/CLIP）。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。左：『手工（過時）』標題，人手設計圖示，下方三點『無語意、敏感、無遷移』。右：『ViT/CLIP』標題，神經網絡圖示，下方三點『語意、穩健、通用』。中間大箭頭『主流趨勢』。白背景、粉彩配色。" --name 01_classical_image_features_fig3_feature_evolution --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
