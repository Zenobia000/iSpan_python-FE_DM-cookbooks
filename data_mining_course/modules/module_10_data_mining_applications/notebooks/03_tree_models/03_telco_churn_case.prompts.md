# 🎨 概念示意圖提示詞 — 電信客戶流失預測

> 對應 notebook：`03_telco_churn_case.ipynb`（模組 M10 · 資料探勘應用）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：客戶流失業務背景 / 資料特徵與預處理流程 / 模型評估與業務應用
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_10_data_mining_applications/notebooks/03_tree_models
```

---

### 圖 1 · 電信客戶流失的商業背景與價值

目的：展示客戶流失對企業的影響、預測的重要性，以及流失客戶復留的經濟價值。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示一個漏斗形流程。『現有客戶』→『預測風險』→分支『干預復留』『流失』。右側標籤『預測』『防範』『挽留』。柔和粉彩配色、白色背景、infographic 風格。" --name 03_telco_churn_case_fig1_business --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 數據特徵與預處理流程

目的：展示電信客戶資料的混合型特徵（數值與類別）及其標準的預處理步驟。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右的流程。『原始特徵』分『數值型』『類別型』；中間『清洗編碼標準化』；右側『準備資料』。簡潔標籤。柔和粉彩配色、白色背景、infographic 風格。" --name 03_telco_churn_case_fig2_preprocessing --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 模型評估指標與業務應用

目的：展示分類模型的多維評估指標（準確率、精確度、召回率、F1、AUC）及其在流失預測中的應用。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示評估儀表板。上方混淆矩陣；下方四個指標『精確度』『召回率』『F1』『AUC』；右側應用『挽留』『配置』『優化』。柔和粉彩配色、白色背景、infographic 風格。" --name 03_telco_churn_case_fig3_evaluation --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
