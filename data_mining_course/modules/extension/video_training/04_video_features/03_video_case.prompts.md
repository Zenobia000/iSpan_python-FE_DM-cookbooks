# 🎨 概念示意圖提示詞 — 動作辨識推論案例（VideoMAE）

> 對應 notebook：`03_video_case.ipynb`（模組 M09 · 多模態特徵 — 影片與跨模態整合）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：端到端推論流程 / VideoMAE 架構與特性 / 預訓練與微調遷移
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/04_video_features
```

---

### 圖 1 · 影片動作辨識端到端流程
目的：展示從影片檔到最終動作預測的完整推論管線。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右流程圖。MP4 檔案 → 影格堆 → 16 幀 → 張量 (T,C,H,W) → VideoMAE → 動作預測。箭頭標註『解碼』『抽樣』『編碼』『推論』。柔和粉彩、白色背景、infographic 風格。" --name 03_video_case_fig1_pipeline --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · VideoMAE 架構與時空注意力
目的：呈現 VideoMAE 如何同時建模空間與時間關係。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示 VideoMAE 架構。左側輸入張量 (N,T,C,H,W)；中央 Transformer 方塊；標註『時間注意力』『空間注意力』；右側輸出 logits。柔和粉彩、白色背景、乾淨教學圖表。" --name 03_video_case_fig2_videomae_arch --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 預訓練與微調的遷移學習
目的：說明 VideoMAE 如何從大規模 Kinetics-400 預訓練轉移到自訂任務。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右遷移學習流程。左『Kinetics-400 預訓練』；中『VideoMAE 特徵』；右『下游任務』。箭頭標註『預訓練』『微調』。柔和粉彩、白色背景、infographic 風格。" --name 03_video_case_fig3_transfer_learning --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
