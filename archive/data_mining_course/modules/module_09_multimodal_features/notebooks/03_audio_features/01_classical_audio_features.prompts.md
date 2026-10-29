# 🎨 概念示意圖提示詞 — 經典音訊特徵

> 對應 notebook：`01_classical_audio_features.ipynb`（模組 M09 · 多模態特徵（音訊））
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：波形→頻譜流程 / MFCC 與手工特徵 / 手工特徵 vs 預訓練模型對比
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/03_audio_features
```

---

### 圖 1 · 波形到頻譜：STFT 的時間–頻率轉換
目的：理解「一維波形」如何透過 STFT 轉成「時間×頻率」的頻譜圖。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。左：波形線，標『波形』。中：窗口圖示，標『短時窗口』。右：彩色頻譜熱圖，橫軸『時間』縱軸『頻率』。標籤清晰大字。粉彩配色、白色背景。" --name 01_classical_audio_features_fig1_stft --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · MFCC 與手工頻譜特徵
目的：展示 MFCC 和常見音色特徵（spectral centroid、rolloff、ZCR）如何從頻譜提取。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。梯形漏斗圖由上而下：頂『頻譜圖』。中層四個特徵『MFCC』『Centroid』『Rolloff』『ZCR』。底『特徵向量』。標籤清晰大字。粉彩配色、白色背景。" --name 01_classical_audio_features_fig2_features --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 手工特徵 vs 預訓練模型對比
目的：說明為何 2026 主流改用預訓練音訊模型，而非手工特徵。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。左欄『手工特徵』人工設計、低語意。右欄『預訓練模型』自學、高語意、多任務。中間標『趨勢：預訓練取代手工』。對比風格。標籤清晰大字。粉彩配色、白色背景。" --name 01_classical_audio_features_fig3_comparison --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
