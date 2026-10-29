# 🎨 概念示意圖提示詞 — 音訊下游與語音辨識

> 對應 notebook：`04_audio_downstream.ipynb`（模組 M11 · 大模型資料前處理與訓練）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：Whisper ASR 推論流程 / wav2vec2 微調管線 / 音訊資料格式對比
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_11_large_model_training/notebooks
```

---

### 圖 1 · Whisper 語音辨識推論流程

目的：展示從音檔讀入、音訊前處理，到 Whisper 模型輸出文字轉錄的整條線。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右的流程圖。第一格『音訊檔案』MP3/WAV 圖示→箭頭『讀檔+重取樣 16kHz』→『波形向量』；第二格『Whisper 模型』編碼器解碼器圖示；箭頭『推論』→『轉錄文字』；第三格『輸出結果』與信心分數。柔和粉彩配色、白色背景、資訊圖表風格。" --name 04_audio_downstream_fig1_whisper_inference --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · wav2vec2 音訊分類微調

目的：展示 wav2vec2 如何從原始波形直接學習分類特徵，微調過程與評估。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示 wav2vec2 微調管線。上方『訓練資料』：音訊檔案+標籤；中上『波形前處理』16kHz 標籤；中間『wav2vec2 模型』架構圖；下方『Trainer 訓練』Loss 曲線；最下『評估指標』（準確率、混淆矩陣）。柔和粉彩配色、白色背景、乾淨教學圖表。" --name 04_audio_downstream_fig2_wav2vec2_finetune --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · Whisper 與 wav2vec2 任務對比

目的：直覺區分兩個模型的不同角色：「語音轉文字」vs「音訊分類」。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，並排對比。左側『Whisper 語音辨識』音訊→『文字輸出』『ASR 任務』；右側『wav2vec2 音訊分類』音訊→『分類標籤』『分類任務』。各側簡述輸入輸出。柔和粉彩配色、白色背景、乾淨對比圖表。" --name 04_audio_downstream_fig3_whisper_vs_wav2vec2 --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
