# 🎨 概念示意圖提示詞 — 缺失值概述與視覺化

> 對應 notebook：`01_missing_data_overview.ipynb`（模組 M03 · 缺失值與異常值）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：缺失類型分類 / 缺失視覺化套件 / 缺失模式識別
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_03_missing_outliers/notebooks
```

---

### 圖 1 · 缺失類型三分法：MCAR、MAR、MNAR

目的：一眼看懂缺失值成因分類與其對分析的影響。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，三欄並排對比。左欄標題『MCAR 隨機缺失』，顯示均勻分佈的缺失點，代表完全隨機；中欄標題『MAR 隨機缺失』，顯示與其他變數相關的缺失模式，用箭頭指向因果關係；右欄標題『MNAR 非隨機缺失』，顯示聚集在某值以上或以下的缺失，代表系統性缺失。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 01_missing_data_overview_fig1_types --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · missingno 視覺化套件展示

目的：展示矩陣圖、樹狀圖、熱力圖三種視覺化方式如何揭露缺失模式。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右三個視覺化面板。左邊為矩陣圖 matrix，顯示多列資料框，其中某些列有白色缺失指示符；中間為樹狀圖 dendrogram，顯示縱橫兩軸樹狀結構；右邊為熱力圖 heatmap，顯示相鄰變數缺失的相關性用色塊表示。上方標註繁體中文『missingno 視覺化』與『識別缺失模式』。柔和粉彩配色、白色背景、教學圖表風格、繁體中文標籤清晰。" --name 01_missing_data_overview_fig2_missingno --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 缺失處理決策樹：評估與選擇

目的：引導分析師根據缺失程度與模式選擇適當的策略。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，從頂端「缺失值」開始的決策樹流程。第一級分支：『缺失率超過 50%？』左邊『是 → 刪除欄位』、右邊『否』；第二級分支：『MNAR 非隨機？』左邊『是 → 需特殊處理』、右邊『否 → 可插補或刪除列』。各分支路徑用彩色箭頭連接，終點為決策框標註繁體中文『刪除』『插補』『保留』。柔和粉彩配色、白色背景、流程圖風格、繁體中文標籤清晰。" --name 01_missing_data_overview_fig3_decision_tree --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
