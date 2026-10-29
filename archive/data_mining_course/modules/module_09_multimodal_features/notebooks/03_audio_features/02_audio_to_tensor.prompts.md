# 🎨 概念示意圖提示詞 — 波形到張量前處理

> 對應 notebook：`02_audio_to_tensor.ipynb`（模組 M09 · 多模態特徵（音訊））
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：三步前處理流程 / 波形與頻譜的資料結構對照 / 批次化張量形狀
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/03_audio_features
```

---

### 圖 1 · 音訊前處理三步驟
目的：一眼看清「重採樣→單聲道→正規化」的標準前處理管線。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。左至右三步流程：①多采樣率波形『重採樣16kHz』。②雙聲道『轉Mono』。③居中波形『正規化』。步驟間右箭頭連結。標籤清晰大字。粉彩配色、白色背景。" --name 02_audio_to_tensor_fig1_preprocessing --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 波形與 log-mel 頻譜的資料結構
目的：說明「波形 (N, samples)」與「頻譜 (N, n_mels, T)」兩種表示的維度與含義。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。上列『波形Waveform』展示N=3波形並排，標『samples』。下列『Log-Mel頻譜』展示彩色頻譜並排，標『n_mels, T』。中間雙向箭頭表轉換。標籤清晰大字。粉彩配色、白色背景。" --name 02_audio_to_tensor_fig2_structures --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 批次化張量形狀與 Padding
目的：展示不同長度音訊如何透過 padding 統一成批次張量。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。左邊三個不同長度波形『短』『中』『長』。箭頭指右『Pad等長』。右邊堆疊矩形張量，標『批次(N, samples)』。淺灰表padding。標籤清晰大字。粉彩配色、白色背景。" --name 02_audio_to_tensor_fig3_batching --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
