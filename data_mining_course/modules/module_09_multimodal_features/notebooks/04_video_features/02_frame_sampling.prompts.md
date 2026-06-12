# 🎨 概念示意圖提示詞 — 影格抽樣策略與前處理

> 對應 notebook：`02_frame_sampling.ipynb`（模組 M09 · 多模態特徵 — 影片與跨模態整合）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：三種抽樣策略比較 / 抽樣選擇與權衡 / Clip vs Frame 處理方式
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/04_video_features
```

---

### 圖 1 · 三種影格抽樣策略對比
目的：視覺展示均勻、密集、分段三種抽樣在時間軸上的不同模式。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，三行時間軸對比圖。上方時間軸 0-T，下方標記點。第一行『均勻』；第二行『密集』；第三行『分段』。柔和粉彩、白色背景、乾淨 infographic 風格。" --name 02_frame_sampling_fig1_strategies --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 影格數量與特性權衡
目的：展示不同影格數量在「時間代表性」與「計算成本」間的權衡。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，二維座標圖。橫軸『成本』；縱軸『時間代表性』。四個節點『4幀』『8幀』『16幀』『32幀』沿對角線。柔和粉彩、白色背景、乾淨教學圖表。" --name 02_frame_sampling_fig2_tradeoff --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · Clip-level vs Frame-level 處理方式
目的：對比兩種影片處理架構：整段序列 vs 逐幀獨立處理。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，左右兩列對比。左『Clip-level』多幀進一個 Transformer；右『Frame-level』各幀獨立編碼。標註『時間建模』與『逐幀處理』。柔和粉彩、白色背景、乾淨 infographic 風格。" --name 02_frame_sampling_fig3_clip_vs_frame --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
