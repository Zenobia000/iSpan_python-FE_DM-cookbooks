# 🎨 概念示意圖提示詞 — 現代影像表示（ViT 與 CLIP）

> 對應 notebook：`03_modern_image_representations.ipynb`（模組 M09 · 多模態特徵（圖像））
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：ViT 的 Patch 分割機制 / ViT 特徵抽取流程 / CLIP 的圖文對齊
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/02_image_features
```

---

### 圖 1 · Vision Transformer（ViT）的 Patch 分割
目的：展示 ViT 如何將影像分割成「視覺 token」序列。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。中央彩色貓咪圖 224×224，用網格線分成 14×14 方格。右側箭頭指向向量序列。標籤『224×224』『16×16 Patch』『196 Token』『+[CLS]→197』『進 Transformer』。白背景、粉彩配色。" --name 03_modern_image_representations_fig1_vit_patches --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · ViT 特徵抽取流程（用 timm）
目的：展示從影像到固定維度特徵向量的完整轉換。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。左至右四步：(N,3,224,224)『批次』→ 『Patch 分割』→ (N,197,D)『197 Token』『D=768』→ 『Transformer』→ [CLS]『提取特徵』→ (N,768)『特徵向量』『下游用』。白背景、粉彩配色。" --name 03_modern_image_representations_fig2_vit_pipeline --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · CLIP：影像-文字對齊與 Zero-shot 分類
目的：展示 CLIP 如何將影像與文字嵌入同一向量空間，實現無需訓練的分類。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。中央大圓代表統一向量空間。左：貓咪圖 → 『圖像編碼器』→ 圓內。右：『貓咪』『狗』『鳥』文字 → 『文本編碼器』→ 圓內。貓圖與貓文靠近，其他遠。下方箭頭『選最相似類別』。白背景、粉彩配色。" --name 03_modern_image_representations_fig3_clip_alignment --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
