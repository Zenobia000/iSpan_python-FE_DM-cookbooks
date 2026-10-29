# 🎨 概念示意圖提示詞 — 影片解碼與張量表示

> 對應 notebook：`01_video_to_tensor.ipynb`（模組 M09 · 多模態特徵 — 影片與跨模態整合）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：影片解碼流程 / 張量維度演進 / 任務標籤格式
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/04_video_features
```

---

### 圖 1 · 影片解碼流程：從檔案到影格序列
目的：一眼看懂影片檔是如何解碼成影格序列的。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右三格流程圖。左格 MP4 檔案；中格堆疊影格標註『T 張影格』；右格排列張量標註『(T,C,H,W)』。箭頭標註『解碼』『排列』。柔和粉彩、白色背景、乾淨 infographic 風格。" --name 01_video_to_tensor_fig1_decode_flow --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 張量維度階梯：單影格到批次
目的：呈現從二維影像 (H, W) 逐步演進到五維批次 (N, T, C, H, W) 的維度變化。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由下至上的階梯圖。底層『影格 (H,W)』；次層『RGB (C,H,W)』；中層『序列 (T,C,H,W)』；頂層『批次 (N,T,C,H,W)』。不同色彩區分各層，白色背景、infographic 風格。" --name 01_video_to_tensor_fig2_tensor_dims --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 影片任務標籤格式對比
目的：展示不同影片任務（動作分類、時序定位、字幕）的標籤結構差異。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，三欄並排。左欄『分類』標註『(N,)』；中欄『時序定位』標註『時間段』；右欄『字幕』標註『文本』。柔和粉彩、白色背景、乾淨教學圖表。" --name 01_video_to_tensor_fig3_label_formats --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
