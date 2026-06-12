# 🎨 概念示意圖提示詞 — 案例實戰 Insurance 資料集

> 對應 notebook：`04_insurance_case.ipynb`（模組 M05 · 特徵縮放與變換）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：特徵分佈概覽 / 分策略選擇 / 處理前後對比
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_05_scaling_transformation/notebooks
```

---

### 圖 1 · Insurance 資料集特徵分佈概覽

目的：展示四個主要特徵（age、bmi、children、charges）的原始分佈特徵。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，四個並排的直方圖展示 Insurance 資料集的四個特徵。第一個『age』呈現均勻分佈，標註『年齡』『均勻分佈』；第二個『bmi』呈現典型常態分佈，標註『身體質量指數』『常態分佈』；第三個『children』為離散柱狀，標註『子女數』『計數變數』；第四個『charges』為高度右偏分佈，標註『醫療費用』『高度右偏』『需轉換』。柔和粉彩配色、白色背景、X軸Y軸標籤清晰、繁體中文標籤完整。" --name 04_insurance_case_fig1_distribution_overview --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 分策略選擇：根據分佈特性決定預處理方法

目的：決策樹式展示如何根據特徵的分佈形狀與偏態程度選擇合適的預處理策略。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，一個決策樹結構，頂部為『特徵評估』；第一層分支『偏態嚴重？』分為『是』與『否』；『是』的分支指向『先冪轉換 (Log/Box-Cox)』再指向『再標準化』，標註『例：charges』；『否』的分支直接指向『直接標準化』，標註『例：age, bmi』；下方另一個『有異常值？』分支指向『使用RobustScaler』，標註『例：children』。柔和粉彩配色、白色背景、箭頭清晰、決策點圓形、繁體中文標籤清楚。" --name 04_insurance_case_fig2_strategy_decision --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 預處理前後對比：四大特徵的變化

目的：視覺化展示應用完整預處理流程後，四個特徵從原始到轉換的全程轉變。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，上方為『預處理前』的四個直方圖，分別展示age（均勻）、bmi（常態）、children（離散）、charges（右偏）；下方為『預處理後』的四個直方圖，age與bmi形狀保持但尺度改變（以0為中心），children經RobustScaler縮放，charges經Log轉換+標準化變成對稱的常態分佈，標註繁體中文『預處理前』『預處理後』『形狀vs尺度』。柔和粉彩配色、白色背景、清晰箭頭連接上下對應特徵、Y軸標籤『頻率』。" --name 04_insurance_case_fig3_before_after --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
