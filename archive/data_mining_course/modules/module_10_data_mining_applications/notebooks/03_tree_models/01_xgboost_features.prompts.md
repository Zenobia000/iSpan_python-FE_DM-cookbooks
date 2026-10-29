# 🎨 概念示意圖提示詞 — XGBoost 特徵重要性

> 對應 notebook：`01_xgboost_features.ipynb`（模組 M10 · 資料探勘應用）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：梯度提升樹基本概念 / 特徵重要性三種衡量方式 / 特徵重要性應用
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_10_data_mining_applications/notebooks/03_tree_models
```

---

### 圖 1 · 梯度提升樹核心思想

目的：視覺化梯度提升的「逐樹優化」思想，展示如何透過多個決策樹的組合達到高準確度。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右展示逐步過程。樹1（誤差紅色）→樹2（修正）→樹3（優化）→『樹組合=強模型』。簡潔標籤『樹1』『樹2』『樹3』。柔和粉彩配色、白色背景、infographic 風格。" --name 01_xgboost_features_fig1_concept --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 特徵重要性三種衡量方式

目的：展示 `weight`、`gain`、`cover` 三種特徵重要性衡量方式的定義與含義。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示三個並排的卡片。『Weight』『Gain』『Cover』各顯示定義與視覺化。簡潔標籤。柔和粉彩配色、白色背景、infographic 風格。" --name 01_xgboost_features_fig2_importance_types --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 特徵重要性應用場景

目的：展示特徵重要性在模型解釋、特徵選擇、業務決策中的實際應用。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示一個水平條形圖排序，列出5個特徵如『年齡、金額、餘額、頻率、分數』。右側應用：『解釋』『選擇』『決策』。柔和粉彩配色、白色背景、infographic 風格。" --name 01_xgboost_features_fig3_applications --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
