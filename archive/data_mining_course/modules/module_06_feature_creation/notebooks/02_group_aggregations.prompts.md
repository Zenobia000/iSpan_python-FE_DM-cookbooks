# 🎨 概念示意圖提示詞 — 分組聚合特徵

> 對應 notebook：`02_group_aggregations.ipynb`（模組 M06 · 特徵創造）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：從個體到群體的視角轉變 / groupby 聚合流程 / 多層次特徵合併
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_06_feature_creation/notebooks
```

---

### 圖 1 · 個體 vs 群體視角：從單筆交易到客戶集合統計
目的：展示分組聚合如何將離散的個體觀測轉化為更具洞察力的群體統計。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分為左右兩部分對比。左側顯示多筆銷售交易列表，每筆交易顯示單一客戶購買金額（如客戶A買100、客戶B買200、客戶A又買150），標註繁體中文『個體觀測』『散亂資訊』；箭頭指向右側，右側顯示按客戶聚合後的統計表格，包含各客戶的總購買額、平均購買額、購買次數等，標註繁體中文『群體聚合』『客戶A：總計250、平均125、次數2』『統計洞察』。柔和粉彩配色（藍綠粉）、白色背景、清晰的轉化圖示、繁體中文標籤。" --name 02_group_aggregations_fig1_perspective --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · pandas groupby 與 agg 聚合流程
目的：展示 groupby 分組與多聚合函數應用的機制。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由上至下的流程圖。上方為完整的銷售資料表，包含客戶ID欄、商品類別欄、購買金額欄；中間箭頭標註『groupby(customer_id)』進行分組；隨後展示多個聚合函數（mean、sum、count、std）分別作用在分組資料上，各自產生不同的結果欄位；下方最終輸出聚合結果表格。標註繁體中文『原始資料』『按客戶分組』『應用聚合函數』『均值、總和、計數、標準差』『聚合結果』。柔和粉彩配色、白色背景、infographic 流程圖風格、繁體中文標籤清晰。" --name 02_group_aggregations_fig2_workflow --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 多層次聚合與 merge 合併回原資料
目的：展示如何從多個聚合維度（客戶、類別）創建特徵，並重新整合到原始 DataFrame。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示兩個平行的聚合路徑。左側路徑：原始資料 → groupby(customer_id) → 客戶聚合特徵表；右側路徑：原始資料 → groupby(product_category) → 類別聚合特徵表；兩個聚合特徵表都分別以 merge(left join) 的方式回到中央的原始資料表，形成一個富集後的完整資料表。標註繁體中文『原始交易表』『客戶層級聚合』『商品類別層級聚合』『豐富的上下文資訊』『多維度融合』。柔和粉彩配色、白色背景、資訊融合圖示風格、繁體中文標籤清晰。" --name 02_group_aggregations_fig3_multilevel_merge --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
