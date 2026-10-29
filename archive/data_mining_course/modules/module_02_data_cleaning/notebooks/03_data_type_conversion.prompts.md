# 🎨 概念示意圖提示詞 — 資料型態轉換

> 對應 notebook：`03_data_type_conversion.ipynb`（模組 M02 · 資料清理）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：型態問題的影響 / 轉換方法對比 / 錯誤處理
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_02_data_cleaning/notebooks
```

---

### 圖 1 · 錯誤的型態造成的問題
目的：展示不正確的資料型態（如 object）導致的四大問題。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，中央為一個 DataFrame，某欄被標記為『object』型態。周圍四個警告框分別指向不同問題：『計算錯誤』（無法求和）、『記憶體浪費』（object 佔用空間大）、『模型不相容』（機器學習無法處理）、『分析受限』（無法時間篩選）。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 03_data_type_conversion_fig1_problems --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 三種轉換方法的對比
目的：並排展示 .astype() / pd.to_numeric() / pd.to_datetime() 的用途和流程。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右三列。第一列標題『.astype()』，顯示字串轉為 int、float、category 的簡單轉換；第二列標題『pd.to_numeric()』，顯示含有無效值的字串列，通過 errors='coerce' 參數轉為數值（無效值變 NaN）；第三列標題『pd.to_datetime()』，顯示多種日期格式字串統一轉為日期型態。柔和粉彩配色、白色背景、教學對比圖表、繁體中文標籤清晰。" --name 03_data_type_conversion_fig2_methods --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 錯誤處理與轉換策略
目的：說明如何安全地處理轉換過程中的異常值。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，流程圖展示轉換流程。首先是『原始欄位』混合有效值和無效值；箭頻進入『轉換嘗試』決策節點；分岔為『errors=raise』（發生異常）和『errors=coerce』（無效值→NaN）兩條路；最終都輸出『清理後的資料』。標註繁體中文『raise：拋出異常』『coerce：轉為 NaN』『ignore：回傳原值』。柔和粉彩配色、白色背景、流程圖風格、繁體中文標籤清晰。" --name 03_data_type_conversion_fig3_error_handling --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
