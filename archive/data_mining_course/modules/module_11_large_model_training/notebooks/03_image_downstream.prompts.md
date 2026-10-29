# 🎨 概念示意圖提示詞 — 影像下游微調與 Zero-shot

> 對應 notebook：`03_image_downstream.ipynb`（模組 M11 · 大模型資料前處理與訓練）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：ViT 與 CLIP 對比 / 微調流程 / 訓練資料格式
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_11_large_model_training/notebooks
```

---

### 圖 1 · ViT 微調 vs CLIP Zero-shot 選擇樹

目的：幫使用者決策：何時微調 ViT？何時用 CLIP zero-shot？

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，一棵決策樹。頂部分叉『有標註資料？』；左支『有』→『ViT 微調』，標註『準確率高、成本中等』；右支『無』→『CLIP Zero-shot』，標註『無需訓練、成本低』。各方案下示金字塔圖表示成本遞增。柔和粉彩配色、白色背景、決策圖表風格。" --name 03_image_downstream_fig1_vit_vs_clip --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · ViT 微調完整流程

目的：展示「讀圖 → resize/正規化 → 模型 → Trainer 訓練 → 評估」的全流程。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右的水平流程。第一格『影像+標籤』資料→箭頭『前處理』→『pixel_values』；第二格『ViT 模型』視覺 patch 圖示；箭頭『Trainer 訓練』→『Loss 曲線』；第三格『評估指標』（準確率、混淆矩陣）。柔和粉彩配色、白色背景、資訊圖表風格。" --name 03_image_downstream_fig2_vit_training_flow --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · CLIP Zero-shot 原理與用法

目的：直覺展示 CLIP 如何用文字 prompt 做分類，無需訓練。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示 CLIP 的二元分支。上方『待分類影像』輸入；左下『視覺編碼器』→『影像向量』；上方『文字 prompt』；右下『文本編碼器』→『文字向量』；中央『相似度計算』；底部『預測輸出』。柔和粉彩配色、白色背景、乾淨教學圖表。" --name 03_image_downstream_fig3_clip_zeroshot --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
