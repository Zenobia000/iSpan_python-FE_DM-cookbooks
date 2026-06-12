# 🎨 概念示意圖提示詞 — LightGBM 特徵重要性

> 對應 notebook：`02_lightgbm_features.ipynb`（模組 M10 · 資料探勘應用）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：LightGBM 與 XGBoost 架構對比 / LightGBM 特徵重要性衡量 / 性能與效率優勢
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_10_data_mining_applications/notebooks/03_tree_models
```

---

### 圖 1 · LightGBM vs XGBoost 樹結構對比

目的：視覺化兩種梯度提升框架的核心差異：LightGBM 採用葉子生長、XGBoost 採用層級生長。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分為左右兩欄對比。左『XGBoost』：平衡樹結構；右『LightGBM』：深化樹結構。下方對比『速度、記憶體、過擬合』。柔和粉彩配色、白色背景、infographic 風格。" --name 02_lightgbm_features_fig1_structure --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · LightGBM 特徵重要性衡量方式

目的：展示 LightGBM 中 `split`（分裂次數）和 `gain`（損失減少）兩種重要性衡量方式。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示並排兩個卡片。『Split』『Gain』各顯示定義與圖示。簡潔標籤。柔和粉彩配色、白色背景、infographic 風格。" --name 02_lightgbm_features_fig2_importance --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · LightGBM 性能與效率優勢

目的：展示 LightGBM 相比 XGBoost 在訓練速度、記憶體使用、大數據場景中的優勢。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示三個並排的性能對比圖表。『速度』『記憶體』『大數據』。LightGBM 優於 XGBoost。簡潔標籤。柔和粉彩配色、白色背景、infographic 風格。" --name 02_lightgbm_features_fig3_advantages --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
