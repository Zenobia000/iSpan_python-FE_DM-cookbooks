# 🎨 概念示意圖提示詞 — 環境聲音分類案例

> 對應 notebook：`04_urban_sound_case.ipynb`（模組 M09 · 多模態特徵（音訊））
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：聲音分類管線全景 / MFCC 特徵路線 / 預訓練嵌入路線與性能對比
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/03_audio_features
```

---

### 圖 1 · 聲音分類全流程
目的：從「載入音訊→標準化→提取特徵→分類→評估」的完整管線。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。左至右流程：①載入②16kHz Mono③特徵分叉(MFCC、wav2vec2)④向量⑤分類器⑥評估。標籤清晰大字。粉彩配色、白色背景。" --name 04_urban_sound_case_fig1_pipeline --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 路線 A：MFCC 手工特徵
目的：展示經典路線「頻譜→MFCC 聚合→特徵向量→邏輯回歸」。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。路線A詳細步驟：波形→頻譜→MFCC→統計(均值、標差)→向量→分類。底部標『輕量、可解釋、低語意』。褐/金黃色調。標籤清晰大字。粉彩配色、白色背景。" --name 04_urban_sound_case_fig2_mfcc_route --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 路線 B：預訓練嵌入 vs 路線 A 性能對比
目的：說明「預訓練嵌入+邏輯回歸」如何勝過「MFCC+邏輯回歸」，並引出後續端到端微調。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫。上『MFCC+LogReg』準確率較低。下『wav2vec2嵌入+LogReg』準確率更高。底部標『下一步：端到端微調』。對比風格。標籤清晰大字。粉彩配色、白色背景。" --name 04_urban_sound_case_fig3_comparison --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
