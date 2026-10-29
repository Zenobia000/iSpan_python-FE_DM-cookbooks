# 🎨 概念示意圖提示詞 — Apriori 演算法

> 對應 notebook：`01_apriori_algorithm.ipynb`（模組 M10 · 資料探勘應用）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：關聯規則三項指標 / Apriori 迭代流程 / 商業應用場景
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_10_data_mining_applications/notebooks/01_association_rules
```

---

### 圖 1 · 關聯規則三項核心指標

目的：一眼看懂支持度、置信度、提升度的定義與關係，理解如何度量商品間的購買模式。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示三個並排的卡片，分別標註繁體中文『支持度』『置信度』『提升度』。第一張卡為一個購物籃含牛奶與麵包，強調『支持度』；第二張卡顯示牛奶指向麵包的箭頭，強調『置信度』；第三張卡為兩個數據點的比較，強調『提升度』。柔和粉彩配色、白色背景、清晰的資訊圖表(infographic)風格。" --name 01_apriori_algorithm_fig1_metrics --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · Apriori 演算法迭代流程

目的：展示「頻繁項集生成 → 規則提取」的核心演算邏輯，從一項到多項的迭代過程。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由上至下的垂直流程圖。頂部『原始資料』；中間『篩選、合併、迭代』的循環步驟；底部『生成規則』。用簡潔的標籤『頻繁項集』『規則』標註各階段。柔和粉彩配色、白色背景、infographic 風格。" --name 01_apriori_algorithm_fig2_process --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 商業應用場景對比

目的：展示關聯規則在零售、推薦、促銷等多個實際場景中的應用價值。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，三個並排的場景卡片。左側『商品配置』；中間『推薦系統』；右側『促銷活動』。簡潔標籤，牛奶、麵包等常見商品視覺化。柔和粉彩配色、白色背景、infographic 風格。" --name 01_apriori_algorithm_fig3_applications --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
