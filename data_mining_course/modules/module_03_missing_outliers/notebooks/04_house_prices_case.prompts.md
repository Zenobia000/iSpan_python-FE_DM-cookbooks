# 🎨 概念示意圖提示詞 — 案例實戰：House Prices 資料集

> 對應 notebook：`04_house_prices_case.ipynb`（模組 M03 · 缺失值與異常值）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：完整清理流程 / 異常值移除案例 / 分策略缺失值填補
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_03_missing_outliers/notebooks
```

---

### 圖 1 · 資料清理流程：五步整合工作流

目的：展示從原始資料到清理完成的系統化路徑與檢驗點。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由上至下五個步驟框串聯。步驟一『載入原始資料』含資料表圖示；步驟二『偵測與移除異常值』含箱型圖；步驟三『評估缺失模式』含 missingno 矩陣；步驟四『分策略填補缺失』分為三個分支『MNAR → 填充 None/0』『MAR 類別 → 眾數』『數值 → 中位數』；步驟五『最終檢驗』含完整度確認。每步驟間用彩色箭頭連接，右側標註『品質檢驗點』。柔和粉彩配色、白色背景、流程圖風格、繁體中文標籤清晰。" --name 04_house_prices_case_fig1_workflow --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 異常值案例：GrLivArea 與 SalePrice 的離群點

目的：透過實際案例展示異常值識別與移除的必要性。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，一個二維散點圖。橫軸為『生活面積 GrLivArea』，縱軸為『房價 SalePrice』。大部分點群集中且呈正相關趨勢，用藍色表示。右下角有兩個紅色圓圈標註『異常點：面積大卻價格極低』，用箭頭指向它們。中央用虛線顯示『去除異常值前的迴歸線』傾斜度過陡，『去除異常值後的迴歸線』（綠色）更陡更合理。柔和粉彩配色、白色背景、散點圖風格、繁體中文標籤清晰。" --name 04_house_prices_case_fig2_outlier_example --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 分策略缺失值填補決策與執行

目的：展示根據不同缺失類型選擇不同填補策略的邏輯。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分為三列並排。左列標題『策略一：MNAR』，顯示代表『不存在』的特徵如『Alley 巷道』，用『None』或『0』填充，代表業務邏輯；中列標題『策略二：MAR 類別』，顯示類別欄位如『ZoneClass』，用眾數（最頻繁值）填充，圖示眾多相同值；右列標題『策略三：數值特徵』，顯示 『LotFrontage』，用中位數填充，圖示分佈較離散。各列下方標註『填充方法』與『適用場景』。柔和粉彩配色、白色背景、決策對比圖、繁體中文標籤清晰。" --name 04_house_prices_case_fig3_imputation_strategies --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
