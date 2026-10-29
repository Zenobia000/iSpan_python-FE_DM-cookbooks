# 🎨 概念示意圖提示詞 — 從資料到模型的地圖

> 對應 notebook：`01_data_to_model_map.ipynb`（模組 M11 · 大模型資料前處理與訓練）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：資料模態對應 / 三種訓練範式 / 技術棧生態
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_11_large_model_training/notebooks
```

---

### 圖 1 · 資料模態對應地圖

目的：一眼看懂四種資料模態（文字、圖像、聲音、影片）如何對應到不同的預訓練模型與下游任務。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，一張完整的對應地圖。左側由上到下列出四種資料模態：『文字』、『圖像』、『聲音』、『影片』，各附簡潔圖示。中間欄位列出對應模型：『LLM』、『ViT』、『Whisper』、『VideoMAE』。右側列出典型任務：『分類』、『檢測』、『辨識』、『動作』。流程由左至右的箭頭連接。柔和粉彩配色、白色背景、乾淨資訊圖表風格。" --name 01_data_to_model_map_fig1_modality_map --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 三種訓練範式對比

目的：區分「從零預訓練」、「全參數微調」與「參數高效微調」的算力需求與應用場景。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，三欄對比表。第一欄『預訓練』標註『從零起訓練全部參數、需大量 GPU 與資料』；第二欄『全參數微調』標註『更新全部參數、需多資料高算力』；第三欄『LoRA 微調』標註『低算力、單卡可跑』。各欄頂部用金字塔圖示表示算力需求遞減。柔和粉彩配色、白色背景、資訊圖表風格。" --name 01_data_to_model_map_fig2_training_paradigms --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 2026 訓練技術棧生態

目的：展示核心工具庫（torch、transformers、datasets、peft、trl 等）及其各自的角色。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示 Python 訓練生態。中央為訓練迴圈流程圖：『資料』→『前處理』→『模型』→『訓練』→『評估』。周邊用顏色方框標註工具：『PyTorch』『transformers』『datasets』『PEFT』『TRL』『Accelerate』『Evaluate』。每個方框簡述功能（1-2 字）。柔和粉彩配色、白色背景、乾淨教學圖表。" --name 01_data_to_model_map_fig3_tech_stack --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
