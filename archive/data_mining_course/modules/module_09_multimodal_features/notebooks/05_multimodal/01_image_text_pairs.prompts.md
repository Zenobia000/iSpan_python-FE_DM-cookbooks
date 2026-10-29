# 🎨 概念示意圖提示詞 — 圖文配對與多模態大模型

> 對應 notebook：`01_image_text_pairs.ipynb`（模組 M09 · 多模態特徵 — 影片與跨模態整合）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：CLIP 對比學習對齊 / 圖文配對與對話資料結構 / 多模態編碼流程
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/05_multimodal
```

---

### 圖 1 · CLIP 對比學習：圖文同一向量空間
目的：視覺展示正配對圖文靠近、負配對遠離的學習目標。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，向量空間圓圈。花朵圖像與對應文字用綠線靠近；其他文字用紅線推遠。標註『正配對靠近』『負配對推遠』。柔和粉彩、白色背景、乾淨 infographic 風格。" --name 01_image_text_pairs_fig1_clip_alignment --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 兩種多模態資料結構對比
目的：區分圖文配對與帶圖對話的資料格式。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，左右兩列對比。左『圖文配對』圖加文本『CLIP』；右『帶圖對話』user 問題加圖、assistant 回答『VLM』。柔和粉彩、白色背景、乾淨教學圖表。" --name 01_image_text_pairs_fig2_data_formats --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 多模態編碼流程：視覺與文字
目的：展示圖像和文本如何被各自編碼成向量或 token，再整合進模型。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由上至下流程。左『圖像』→『視覺編碼器』→『視覺 token』；右『文字』→『文字編碼器』→『文字 token』；兩者匯流進『融合器』；輸出『檢索/推論/生成』。柔和粉彩、白色背景、infographic 風格。" --name 01_image_text_pairs_fig3_multimodal_encoding --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
