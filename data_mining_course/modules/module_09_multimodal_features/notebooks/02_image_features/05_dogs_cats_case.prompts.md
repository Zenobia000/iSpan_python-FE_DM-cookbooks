# 🎨 概念示意圖提示詞 — Dogs vs Cats 案例

> 對應 notebook：`05_dogs_cats_case.ipynb`（模組 M09 · 多模態特徵（圖像））
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：CLIP Zero-shot 分類流程 / 凍結 ViT 特徵 + 分類器流程 / 兩條路線對比
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/02_image_features
```

---

### 圖 1 · CLIP Zero-shot 分類流程
目的：展示如何用 CLIP 完全不訓練就進行分類。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。左：貓/狗影像『輸入』。中左：『CLIP 圖像編碼器』。中右：『貓咪、狗』文字『CLIP 文本編碼器』。中央大圓『相似度比較』。右：輸出『預測：貓/狗』『無需訓練』。白背景、粉彩配色。" --name 05_dogs_cats_case_fig1_clip_zeroshot --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 凍結 ViT 特徵 + 分類器訓練流程
目的：展示特徵提取和輕量訓練的兩階段方式。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。左至右：(N,C,H,W)『影像批次』→ 『ViT 特徵（凍結）』❄️ → (N,768)『固定特徵』→ 『訓練分類器』🔥 → 『狗/貓』『少量標註』。白背景、粉彩配色。" --name 05_dogs_cats_case_fig2_frozen_vit_classifier --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 兩條路線的對比與延伸
目的：幫助讀者理解兩種方法的優劣與進階選項。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。左『路線 A：CLIP Zero-shot』列點『無訓練、改類別易、速快』。右『路線 B：凍結 ViT』列點『輕量訓練、需標註、準確高』。下方『進階：ViT 微調（Module 11）』指向上方。白背景、粉彩配色。" --name 05_dogs_cats_case_fig3_pipeline_comparison --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
