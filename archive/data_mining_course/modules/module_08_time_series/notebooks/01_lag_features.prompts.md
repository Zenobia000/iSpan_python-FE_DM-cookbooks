# 🎨 概念示意圖提示詞 — 滯後特徵 (Lag Features)

> 對應 notebook：`01_lag_features.ipynb`（模組 M08 · 時間序列特徵工程）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：滯後特徵概念結構 / 時間依賴性與自相關 / shift() 方法的實作效果
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_08_time_series/notebooks
```

---

### 圖 1 · 滯後特徵的核心概念：時間遞移
目的：展示如何通過過去時間點的觀測值作為當前時間點的新特徵。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由上至下展示時間序列的滯後特徵構造。上方為一條時間序列圖，橫軸標註繁體中文『時間』，縱軸標註『觀測值』，展示 5 個時間點 t=0,1,2,3,4 的數值點。中間展示垂直轉換，顯示原始特徵『Y(t)』與兩個滯後特徵『Y(t-1)滯後1期』『Y(t-2)滯後2期』如何並排排列成新表格。下方為完整的特徵表格，四欄五列，分別標註繁體中文『時間點』『Y(t)目標值』『Y(t-1)滯後1期』『Y(t-2)滯後2期』，清晰展示數值在表格中的排列。柔和粉彩配色、白色背景、infographic 風格。" --name 01_lag_features_fig1_concept --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 時間依賴性與自相關性
目的：視覺化展示時間序列中當前值與過去值的相關性。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示時間序列的自相關性概念。左側為一條波動的時間序列曲線，標註繁體中文『原始時間序列』，顯示明顯的上升下降趨勢。右側展示兩個散點圖並排：上面的散點圖標註繁體中文『滯後1期 高相關性 r≈0.8』，點形成明顯向上趨勢的線性關係；下面的散點圖標註繁體中文『滯後7期 中等相關性 r≈0.5』，點的分散度較大。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 01_lag_features_fig2_autocorr --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · pandas shift() 方法的實作效果
目的：展示如何使用 shift() 創建滯後特徵，以及缺失值的產生。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示 pandas shift() 操作的三個步驟。左側為一個 3 欄 6 列的 DataFrame，第一欄為『索引』，第二欄標註繁體中文『原始值』含 6 個數值 100,105,110,108,115,120。中間顯示箭頭與代碼片段 'df.shift(1)'。右側為結果 DataFrame，展示三欄：『原始值』『移動1期 shift(1)』『移動2期 shift(2)』，清晰顯示數值向下遞移與頂部產生繁體中文『NaN 缺失值』的現象。柔和粉彩配色、白色背景、清晰的資訊圖表、繁體中文標籤。" --name 01_lag_features_fig3_shift_operation --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
