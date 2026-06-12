# 🎨 概念示意圖提示詞 — 交互特徵入門

> 對應 notebook：`01_interaction_features.ipynb`（模組 M06 · 特徵創造）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：交互特徵的本質 / PolynomialFeatures 自動生成 / 領域知識手動創建
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_06_feature_creation/notebooks
```

---

### 圖 1 · 交互特徵的本質：特徵組合捕捉非線性關係
目的：直觀理解為什麼需要交互特徵——單一特徵無法表達組合的力量。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示交互特徵的核心概念。左上顯示兩個獨立的特徵A和B各自繪製為簡單的線性圖表，標註繁體中文『特徵A』『特徵B』『線性關係』；箭頭指向右下，展示它們組合後的二維平面圖，其中曲線顯示非線性關係，標註繁體中文『交互項 A×B』『捕捉非線性』『提升預測力』。柔和粉彩配色（藍綠粉黃）、白色背景、infographic 風格、繁體中文標籤清晰。" --name 01_interaction_features_fig1_essence --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · PolynomialFeatures 自動生成流程
目的：展示 sklearn 工具如何自動膨脹特徵空間，生成多項式與交互項。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由上至下的流程圖。最上方為兩欄的原始特徵表格，標註繁體中文『原始特徵 A、B』；箭頭下方標註『degree=2, interaction_only=False』；中間展示轉換過程，生成五列的擴展特徵表格，標註繁體中文『1（偏差）、A、B、A²、B²、A×B』；下方為結果表格顯示六列。柔和粉彩配色、白色背景、清晰的資訊流轉圖示、繁體中文標籤。" --name 01_interaction_features_fig2_polynomial --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 領域知識 vs 自動生成：手動創建有意義的交互特徵
目的：對比自動化與手工設計，強調領域知識的價值。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分為左右兩側對比。左側標題『自動化工具 PolynomialFeatures』，顯示多個特徵列表（含冗餘項），標註繁體中文『快速便捷、特徵眾多、解釋性弱』；右側標題『領域知識手動創建』，顯示精選的幾個特徵（如房間數/面積、收入/人口、單價×購買量），標註繁體中文『深層洞察、業務意義強、解釋性佳』。中間為分割線。柔和粉彩配色、白色背景、對比鮮明的 infographic 風格、繁體中文標籤清晰。" --name 01_interaction_features_fig3_manual_vs_auto --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
