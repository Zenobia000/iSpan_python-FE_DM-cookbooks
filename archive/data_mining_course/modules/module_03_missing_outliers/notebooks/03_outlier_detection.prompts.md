# 🎨 概念示意圖提示詞 — 異常值偵測方法

> 對應 notebook：`03_outlier_detection.ipynb`（模組 M03 · 缺失值與異常值）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：異常值定義與影響 / IQR 與箱型圖 / Z-score 方法
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_03_missing_outliers/notebooks
```

---

### 圖 1 · 異常值的危害：統計量與模型擬合的扭曲

目的：展示單一異常值如何影響均值、迴歸線及模型學習。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，左右對比。左側標題『無異常值』，顯示一個點群集中且對稱的散點圖，迴歸線平穩、均值線清晰；右側標題『含異常值』，同樣散點圖但右上方有一個紅色離群點，迴歸線被拉扯扭曲、均值線偏移。用橘色箭頭標註『均值被拉高』『迴歸線扭曲』。柔和粉彩配色、白色背景、教學對比圖、繁體中文標籤清晰。" --name 03_outlier_detection_fig1_impact --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 四分位距法（IQR）與箱型圖

目的：闡釋 IQR 法則與箱型圖的幾何意義與異常值判定邊界。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，頂部為數軸，標示最小值、Q1、中位數 Q2、Q3、最大值五個位置。中部為箱型圖，用紫色方框表示 IQR（Q1 到 Q3），中間線為中位數，兩端須鬚延伸到合法邊界『Q1 - 1.5×IQR』與『Q3 + 1.5×IQR』；超出邊界的點用紅色圓圈標示為『異常值』。下方用公式框標註『IQR = Q3 - Q1』與邊界計算式。柔和粉彩配色、白色背景、統計圖表風格、繁體中文標籤清晰。" --name 03_outlier_detection_fig2_iqr_boxplot --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · Z-Score 方法：標準差單位的距離判定

目的：展示 Z-score 如何量化一個點與平均值的偏差，以及異常值閾值。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，呈現一條鐘形常態分佈曲線，橫軸為數據值，縱軸為頻率。標註繁體中文『μ 平均值』於中心、『σ 標準差』為單位寬度。用顏色區塊標示：中央『±1σ 內 68%』綠色、『±2σ 內 95%』黃色、『±3σ 邊界外』紅色標示『異常值（|Z| > 3）』。下方公式框標註『Z = (x - μ) / σ』。柔和粉彩配色、白色背景、統計分佈圖、繁體中文標籤清晰。" --name 03_outlier_detection_fig3_zscore --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
