# 🎨 概念示意圖提示詞 — 購物中心客戶分群

> 對應 notebook：`03_mall_customers_case.ipynb`（模組 M10 · 資料探勘應用）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：客戶資料特徵 / 資料預處理流程 / 客戶分群結果與應用
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_10_data_mining_applications/notebooks/02_clustering
```

---

### 圖 1 · 購物中心客戶資料特徵

目的：展示客戶資料集的主要維度：人口統計（年齡、性別）與消費行為（年收入、支出分數）。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示資料卡片。『年齡』『性別』『收入』『支出分數』四個主要欄位。不同顏色區分，顯示範例資料。簡潔標籤。柔和粉彩配色、白色背景、資訊圖表(infographic)風格。" --name 03_mall_customers_case_fig1_features --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 資料預處理流程

目的：展示從原始資料到準備好進行聚類的完整預處理步驟：特徵選擇、缺失值、標準化。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右的水平流程。『原始資料』→『篩選特徵』→『標準化』→『準備完成』。各步驟用箭頭連接。簡潔標籤。柔和粉彩配色、白色背景、infographic 風格。" --name 03_mall_customers_case_fig2_preprocessing --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 客戶分群結果與行銷應用

目的：展示 K-Means 聚類結果的四到五個客戶群體及其特徵，以及對應的行銷策略。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示一個二維散點圖，X軸年收入，Y軸支出分數，用四種顏色表示四個客戶群體。右側應用：『精準行銷』『等級制度』『推薦』。柔和粉彩配色、白色背景、infographic 風格。" --name 03_mall_customers_case_fig3_segments --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
