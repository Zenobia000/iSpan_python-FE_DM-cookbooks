# 🎨 概念示意圖提示詞 — 異常值對特徵縮放的影響

> 對應 notebook：`03_outliers_impact.ipynb`（模組 M05 · 特徵縮放與變換）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：異常值的破壞性 / 三種縮放器對比 / RobustScaler 原理
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_05_scaling_transformation/notebooks
```

---

### 圖 1 · 異常值對 MinMaxScaler 的災難性影響

目的：直觀展示單一極端異常值如何摧毀歸一化結果，壓縮正常資料。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，上方展示原始資料分佈，大多數資料點集中在0-150區間，突然有一個『500』的異常值（用紅色×標記）；中間標註『MinMaxScaler: (x-min)/(max-min)』；下方為縮放結果，原本150的最大值被映射到0.2附近，而異常值500被映射到1，導致繁體中文標註『正常資料被壓縮到0-0.2』『內部差異丟失』『災難性結果』。柔和粉彩配色、白色背景、清晰的箭頭與數據標籤。" --name 03_outliers_impact_fig1_minmax_disaster --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 三種縮放器在異常值下的表現對比

目的：並排比較 StandardScaler、MinMaxScaler、RobustScaler 的穩健性。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，三個並排的直方圖展示同組含異常值的資料經三種縮放器處理後的分佈。左圖『StandardScaler』，資料大致呈現左移的常態分佈，標註『使用均值/標差』『中度受影響』；中圖『MinMaxScaler』，大部分資料被壓縮到0-0.2，標註『使用最大/最小』『嚴重受影響』『不推薦』；右圖『RobustScaler』，資料分佈保持完整，異常值遠離中心，標註『使用中位數/IQR』『穩健性最佳』『推薦』。柔和粉彩配色、白色背景、Y軸標籤『頻率』、繁體中文標籤清晰。" --name 03_outliers_impact_fig2_three_scalers --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · RobustScaler 的穩健性原理

目的：解釋為何 RobustScaler 對異常值不敏感，以及中位數與 IQR 的關鍵角色。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示一組資料的箱線圖(box plot)，標註繁體中文『中位數』『第一四分位數(Q1)』『第三四分位數(Q3)』『四分位距(IQR)=Q3-Q1』『異常值』，並在旁邊列出『RobustScaler公式』『(x-中位數)/IQR』，強調『不受極端值影響』『中位數穩定』『IQR穩定』。柔和粉彩配色、白色背景、箱線圖清晰、異常值用紅色標記、公式框醒目。" --name 03_outliers_impact_fig3_robust_principle --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
