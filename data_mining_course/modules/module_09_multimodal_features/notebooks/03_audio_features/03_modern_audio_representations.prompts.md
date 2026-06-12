# 🎨 概念示意圖提示詞 — 現代音訊表示

> 對應 notebook：`03_modern_audio_representations.ipynb`（模組 M09 · 多模態特徵（音訊））
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：AutoFeatureExtractor 流程 / Whisper vs wav2vec2 輸入對比 / 音訊嵌入提取與下游任務
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/03_audio_features
```

---

### 圖 1 · AutoFeatureExtractor：模型配套的自動前處理
目的：說明與文字 tokenizer、影像 processor 對應的音訊 feature extractor 概念。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。三欄並排：『文字』Tokenizer、『影像』Processor、『音訊』FeatureExtractor。底部標『統一介面：自動選擇前處理』。標籤清晰大字。粉彩配色、白色背景。" --name 03_modern_audio_representations_fig1_extractor --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · Whisper vs wav2vec2：兩種輸入路線
目的：對比「頻譜輸入」(Whisper) vs「波形輸入」(wav2vec2)，及其典型輸出形狀。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。上列『Whisper頻譜輸入』ASR。下列『wav2vec2波形輸入』表示學習。各列流程：音訊→前處理→模型→輸出。標籤清晰大字。對比風格。粉彩配色、白色背景。" --name 03_modern_audio_representations_fig2_models --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 音訊嵌入提取與下游任務
目的：展示如何從模型的隱藏層取嵌入，再用於下游分類或檢索任務。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。左至右流程：音訊→預訓練模型→隱藏層『(B,T,D)』→池化『(B,D)』→分叉『分類』『檢索』『相似度』。標籤清晰大字。粉彩配色、白色背景。" --name 03_modern_audio_representations_fig3_embeddings --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
