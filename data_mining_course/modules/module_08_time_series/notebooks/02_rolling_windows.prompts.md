# 🎨 概念示意圖提示詞 — 滑動窗口特徵 (Rolling Window Features)

> 對應 notebook：`02_rolling_windows.ipynb`（模組 M08 · 時間序列特徵工程）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：滑動窗口的概念 / 移動平均與波動性 / rolling() 參數與結果
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_08_time_series/notebooks
```

---

### 圖 1 · 滑動窗口的運動機制
目的：展示固定大小的窗口如何在時間序列上逐步滑動，計算統計量。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示滑動窗口在時間序列上的逐步移動。上方為一條時間序列圖，顯示 8 個時間點的數值點。下方分三行展示三個時間步的窗口位置：第一行窗口框住前 3 個點，標註繁體中文『窗口大小=3 平均值=?』；第二行窗口右移一格框住中間 3 個點，標註『窗口移動 計算平均值=?』；第三行窗口再右移框住後 3 個點，標註『逐步滑動 得到時間序列的滑動統計量』。柔和粉彩配色、白色背景、箭頭清晰、infographic 風格。" --name 02_rolling_windows_fig1_sliding --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 移動平均與波動性平滑效果
目的：對比原始時間序列與其移動平均線，展示平滑效果。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，並排展示兩個時間序列圖。左側標註繁體中文『原始時間序列 波動大 有雜訊』，曲線高低起伏明顯。右側標註繁體中文『7日移動平均 平滑趨勢 降低雜訊』，展示更光滑的曲線。上方另加一個文本框標註繁體中文『滑動窗口特徵優點：去雜訊、捕捉局部趨勢、衡量波動性』。柔和粉彩配色、白色背景、兩條曲線對比清晰、infographic 風格。" --name 02_rolling_windows_fig2_smoothing --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · pandas rolling() 方法與參數
目的：展示 rolling() 的常見統計聚合函數與關鍵參數。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示 pandas rolling() 方法的功能結構。中央為一個時間序列數據表格，左右兩側分別展示常見統計函數與參數。左側方框列舉繁體中文『常見聚合函數：mean()平均值、std()標準差、min()最小值、max()最大值、sum()求和』。右側方框列舉繁體中文『重要參數：window 窗口大小、min_periods 最小觀測數、center 居中對齊』。上方顯示代碼片段『.rolling(window=7).mean()』，下方展示結果表格顯示計算結果。柔和粉彩配色、白色背景、清晰的資訊圖表、繁體中文標籤。" --name 02_rolling_windows_fig3_rolling_api --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
