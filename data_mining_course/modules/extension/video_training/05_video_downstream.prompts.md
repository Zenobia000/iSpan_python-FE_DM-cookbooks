# 🎨 概念示意圖提示詞 — 影片下游動作辨識

> 對應 notebook：`05_video_downstream.ipynb`（模組 M11 · 大模型資料前處理與訓練）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：影片資料前處理流程 / VideoMAE 推論 / 微調管線
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_11_large_model_training/notebooks
```

---

### 圖 1 · 影片資料前處理與特徵提取

目的：展示從原始影片到規整張量的前處理步驟：切 clip、抽樣影格、像素化。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右的流程。第一格『原始影片檔案』播放符號→箭頭『分割 clip』→『短片段』；第二格『抽樣影格』展示 16 幀關鍵幀，標註『T=16』；箭頭『正規化』→『像素值』；第三格『規整張量』四維方塊標註『(N,T,C,H,W)』。柔和粉彩配色、白色背景、資訊圖表風格。" --name 05_video_downstream_fig1_preprocessing --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · VideoMAE 推論與預訓練

目的：展示 VideoMAE（在 Kinetics-400 上預訓練）可直接推論 400 種動作。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示推論流程。左側『待分類影片 clip』連續動作影格序列；中間『VideoMAE 模型』方塊；右側『推論輸出』概率排序『跳躍、奔跑、走路』與信心分數。底部標註『400 種動作』。柔和粉彩配色、白色背景、乾淨教學圖表。" --name 05_video_downstream_fig2_videomae_inference --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 影片微調完整管線

目的：展示「自己的動作資料 → 微調 VideoMAE → 新模型」的全流程。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示微調管線。最上方『自訂動作資料集』影片 clip+標籤；中上『前處理』抽樣 16 幀張量化；中間『VideoMAE 基礎模型』更新 num_labels；下方『Trainer 訓練』Loss 曲線與驗證準確率；最下『微調後模型』部署推論。柔和粉彩配色、白色背景、乾淨教學圖表。" --name 05_video_downstream_fig3_videomae_finetune --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
