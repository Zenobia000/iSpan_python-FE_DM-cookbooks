# 🎨 概念示意圖提示詞 — 變數的冪轉換

> 對應 notebook：`02_power_transformations.ipynb`（模組 M05 · 特徵縮放與變換）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：偏態到常態轉換 / Q-Q 圖評估 / 三種方法對比
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_05_scaling_transformation/notebooks
```

---

### 圖 1 · 偏態分佈到常態分佈的轉換

目的：視覺化展示對數轉換如何將右偏分佈矯正為接近常態分佈。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，左邊為一個典型的右偏(正偏)直方圖，資料點集中在左側，長尾向右延伸，標註繁體中文『原始分佈』『右偏 (Skewed)』；中間有一個『log(x)轉換』的箭頭與公式框；右邊為轉換後的直方圖，呈現對稱的鐘形曲線（高斯分佈），標註繁體中文『對數轉換後』『常態分佈 (Normal)』『左右對稱』。柔和粉彩配色、白色背景、直方圖清晰、箭頭與標籤繁體中文清楚。" --name 02_power_transformations_fig1_skewed_to_normal --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · Q-Q 圖：評估常態性

目的：展示如何利用 Q-Q 圖判斷資料是否符合常態分佈。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，並排兩個 Q-Q 圖。左圖為『原始資料』，資料點（藍色圓點）在兩端嚴重偏離紅色對角線，標註繁體中文『原始資料』『偏離對角線』『不符合常態』；右圖為『對數轉換後』，資料點幾乎完美地落在紅色對角線上，標註繁體中文『轉換後資料』『緊貼對角線』『符合常態分佈』。X軸標籤『理論分位數』、Y軸標籤『樣本分位數』、紅色虛線代表完美常態。柔和粉彩配色、白色背景、清晰網格與標籤。" --name 02_power_transformations_fig2_qq_plot --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 三種轉換方法對比

目的：區分對數轉換、Box-Cox、Yeo-Johnson 三種方法的特性與應用範圍。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，三欄並排對比表。第一欄『對數轉換 log(x)』展示一個公式與限制標籤『僅正數』；第二欄『Box-Cox 轉換』展示通用性更強、自動尋找最佳λ的說明，標註『需正數』『自動優化』；第三欄『Yeo-Johnson 轉換』展示最靈活，可處理零與負值，標註『全數字範圍』『最通用』。每欄下方各顯示轉換後的分佈直方圖。柔和粉彩配色、白色背景、表格清晰、繁體中文標籤完整。" --name 02_power_transformations_fig3_methods_comparison --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
