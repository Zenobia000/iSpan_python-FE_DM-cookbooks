# 🎨 概念示意圖提示詞 — 過濾法 (Filter Methods)

> 對應 notebook：`01_filter_methods.ipynb`（模組 M07 · 特徵選擇）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：過濾法核心特點對比 / 三大過濾方法選擇流程 / 統計評估指標對比
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_07_feature_selection/notebooks
```

---

### 圖 1 · 過濾法核心特點：獨立評估不依賴模型

目的：一眼看懂過濾法與其他方法的根本區別——純統計評估，無需訓練任何模型。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示三種特徵選擇方法的並排對比。左側『過濾法 Filter』顯示直接統計指標（相關係數、卡方檢定、F-檢定）評估特徵，無模型圖示，標註繁體中文『獨立評估』『計算快』『無過擬合風險』。中間『包裹法 Wrapper』顯示機器學習模型反覆訓練，標註『依賴模型』『計算慢』。右側『嵌入法 Embedded』顯示模型訓練中提取權重，標註『訓練同步』『自動篩選』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 01_filter_methods_fig1_overview --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 三大過濾方法選擇流程與適用場景

目的：清楚呈現皮爾森相關、卡方檢定、ANOVA F-檢定的適用數據類型與評估流程。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由上至下三欄，分別展示三種過濾法。第一欄『皮爾森相關係數 Pearson Correlation』：輸入為『數值特徵 + 數值目標』箭頭指向『計算相關性 -1 到 +1』輸出『篩選高相關特徵』，視覺化為散點圖趨勢線。第二欄『卡方檢定 Chi-Squared Test』：輸入『類別特徵 + 類別目標』箭頭『計算卡方統計量』輸出『篩選高獨立性特徵』視覺化為列聯表。第三欄『ANOVA F-檢定』：輸入『數值特徵 + 類別目標』箭頭『計算 F-值』輸出『篩選高判別力特徵』視覺化為箱線圖。柔和粉彩配色、白色背景、繁體中文標籤清晰、infographic 風格。" --name 01_filter_methods_fig2_techniques --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 過濾法優缺點與應用決策

目的：總結過濾法的關鍵優點（快速、模型獨立）與缺陷（忽視交互、可能冗餘），助於決策。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分為左右兩側天平秤設計。左側『優點』盤中浮起三張綠卡：『計算速度快』『模型獨立』『無過擬合』，卡片上有對應圖示（計時器、齒輪獨立、盾牌）。右側『缺點』盤中三張紅卡：『忽略特徵交互』『可能選出冗餘』『只看單變量關係』。中央天平指針微傾向左（優點稍多），底部標註繁體中文『適合：初步篩選、高維資料、大規模快速評估』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 01_filter_methods_fig3_tradeoffs --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
