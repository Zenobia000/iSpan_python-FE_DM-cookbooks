# 🎨 概念示意圖提示詞 — 缺失值插補方法

> 對應 notebook：`02_imputation_methods.ipynb`（模組 M03 · 缺失值與異常值）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：插補方法分類 / 單變數 vs 多變數 / KNN 插補原理
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_03_missing_outliers/notebooks
```

---

### 圖 1 · 插補方法全景：複雜度 vs 保真度

目的：展示各種插補技術的複雜度與對原始分佈保護程度的權衡。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，二維散點圖分佈。橫軸為『複雜度』從左至右遞增，縱軸為『對原始分佈的保真度』從下至上遞增。標註五個點位分別為：『刪除法』左下、『均值填充』中左低、『中位數填充』中左低、『KNN 插補』中上、『多重插補』右上。用四邊形框圍繞，標註繁體中文『簡單方法』與『進階方法』。柔和粉彩配色、白色背景、散點圖風格、繁體中文標籤清晰。" --name 02_imputation_methods_fig1_complexity_fidelity --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 單變數 vs 多變數插補對比

目的：直觀展示單變數方法的局限與多變數方法的優勢。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分左右兩區。左側標題『單變數插補』，一個柱狀圖，某列有缺失值，用水平線標籤『均值/中位數/眾數』填充，箭頭指向結果；右側標題『多變數插補（KNN）』，顯示一個資料表，某行有缺失值，用周圍相似列的值加權平均，多個箭頭指向鄰近行。柔和粉彩配色、白色背景、教學對比圖、繁體中文標籤清晰。" --name 02_imputation_methods_fig2_univariate_multivariate --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · KNN 插補原理：尋找相似鄰居

目的：闡釋 K-近鄰插補如何利用資料點間的相似性進行填補。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，一個二維散點圖。中央有一個目標點標註『待插補樣本』用紅色圓圈，周圍有多個黑點代表已完整樣本。用虛線圓圈圈出 K 個最近的鄰近點，用箭頭指向它們，標註繁體中文『尋找 K 個最相似鄰居』；下方顯示一個計算框，標註『權重平均計算』與『加權填補值』。柔和粉彩配色、白色背景、幾何示意圖風格、繁體中文標籤清晰。" --name 02_imputation_methods_fig3_knn_principle --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
