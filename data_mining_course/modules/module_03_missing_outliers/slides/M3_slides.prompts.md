# 🎨 概念示意圖提示詞 — 模組3 缺值與異常值

> 對應講義：`M3_Missing_and_Outliers_Fundamentals.md`
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：缺失機制 / 五種插補的取捨 / 全資料算中位數的後果
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；標籤要完全清晰再改 `--quality high`。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_03_missing_outliers/slides
```

---

### 圖 1 · 缺值可以是訊號
用在第 1 張。目的：讓學員先分清楚缺失機制，再談補值。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，三個並排欄位卡片代表三種缺失機制。左卡『MCAR 完全隨機』，資料表中零星幾格空白且分佈均勻，標註『補值安全』綠勾。中卡『MAR 有條件隨機』，空白集中在某個類別的列上，旁邊一個箭頭指向另一欄，標註『用其他欄位推估』黃色驚嘆號。右卡『MNAR 與自身有關』，高收入那幾列的收入欄全部空白，標註『補了等於捏造，改成加指示欄』紅色叉。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M3_fig1_missing_as_signal --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 五種插補方法的取捨
用在第 3 張。目的：一張圖比完五種方法的代價與洩漏風險。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，一個五欄的比較圖。五欄依序為『整列刪除』『均值』『中位數』『KNN』『加指示欄』。每欄用三個小圖示標示：資料量損失（沙漏）、分佈失真（被壓扁的鐘形曲線）、洩漏風險（放大鏡）。整列刪除的沙漏最滿，均值的鐘形曲線被壓得最扁，KNN 的放大鏡標紅色警告並註明『測試集不能參與鄰居計算』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M3_fig2_imputation_tradeoff --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 用全資料算中位數的後果
用在第 7 張。目的：把這個模組最常見的錯誤畫成前後對照。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，上下兩排流程對照。上排標紅色叉：『整份資料』→『算中位數 28.5』→『切分』→『訓練』，測試分數顯示 0.91，旁邊一個放大鏡偷看測試資料。下排標綠色勾：『整份資料』→『切分』→『只用訓練集算中位數 27.9』→『同一個值填兩邊』，測試分數顯示 0.84 並標註『這個才可信』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M3_fig3_leak_median --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
