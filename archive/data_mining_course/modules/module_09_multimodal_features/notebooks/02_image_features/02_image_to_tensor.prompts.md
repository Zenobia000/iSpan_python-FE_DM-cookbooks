# 🎨 概念示意圖提示詞 — 影像到張量前處理

> 對應 notebook：`02_image_to_tensor.ipynb`（模組 M09 · 多模態特徵（圖像））
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：四步前處理管線 / Channels-first vs Channels-last / 標籤資料結構
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/02_image_features
```

---

### 圖 1 · 標準前處理四步管線
目的：展示從原始檔案到模型輸入的完整轉換流程。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。左至右四步：JPG → uint8 陣列 → Resize/Crop → float32 張量。箭頭上方分別標『解碼』『裁切』『張量』『標準化』。最終張量準備進模型。白背景、粉彩配色。" --name 02_image_to_tensor_fig1_preprocess_pipeline --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · Channels-first 與 Channels-last 對比
目的：澄清 PyTorch 與 TensorFlow/Numpy 的排列慣例差異。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。左：立方體，標『PyTorch』『(N,C,H,W)』『C 在前』。右：同立方體，標『TensorFlow』『(N,H,W,C)』『C 在後』。中間雙向箭頭。下方警告『混淆掉分』。白背景、粉彩配色。" --name 02_image_to_tensor_fig2_channels_format --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 任務級標籤資料結構對比
目的：展示不同視覺任務的標籤組織方式。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。由上至下四種任務：『分類』貓咪圖 → 標『3』；『多標籤』風景圖 → 標『[1,0,1,0]』；『檢測』多邊框 → 標『[x,y,w,h,cls]』；『分割』像素著色 → 標『(H,W)』。白背景、粉彩配色、清晰區隔。" --name 02_image_to_tensor_fig3_label_formats --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
