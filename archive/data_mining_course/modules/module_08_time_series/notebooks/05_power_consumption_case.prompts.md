# 🎨 概念示意圖提示詞 — 綜合案例：電力消耗預測

> 對應 notebook：`05_power_consumption_case.ipynb`（模組 M08 · 時間序列特徵工程）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：時間序列特徵工程的完整流程 / 時間序列交叉驗證 / 模型評估與特徵重要性
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_08_time_series/notebooks
```

---

### 圖 1 · 時間序列特徵工程的完整流程
目的：展示從原始電力消耗數據到特徵工程的完整管道。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右的水平流程圖展示時間序列特徵工程全流程。第一格為一個標註繁體中文『原始時間序列 小時級別電力消耗』的時間序列圖。第二格顯示三個分支，分別標註繁體中文『日期時間特徵 年月日時』『滯後特徵 lag1,lag24,lag168』『滑動特徵 rolling mean, std』。第三格為一個豐富的特徵表格，標註繁體中文『合併特徵集 多維度特徵矩陣』。第四格為模型與預測結果，標註繁體中文『機器學習預測 電力消耗值』。箭頭連接各步驟，柔和粉彩配色、白色背景、infographic 風格。" --name 05_power_consumption_case_fig1_pipeline --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 時間序列交叉驗證：避免數據洩漏
目的：展示正確的時間序列數據分割方法，避免使用未來數據訓練。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示時間序列數據的正確分割方式。上方展示一條長時間序列，標註繁體中文『時間向前進行 →』。下方展示四個分割區間：第一個區間標註繁體中文『Fold 1：訓練 測試』，用不同色塊區分；第二個區間標註『Fold 2：訓練 測試』，訓練集擴大，測試集後移；第三個區間標註『Fold 3：訓練 測試』；第四個區間標註『Fold 4：訓練 測試』。箭頭顯示時間前進方向，清晰表達「只用過去預測未來」的原則。柔和粉彩配色、白色背景、infographic 風格。" --name 05_power_consumption_case_fig2_cv_strategy --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 模型評估與特徵重要性分析
目的：展示時間序列預測模型的評估指標與重要特徵識別。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示模型評估的三個視角。左側為模型評估指標框，標註繁體中文『評估指標 RMSE：0.15 MAE：0.12 R²：0.92』。中央上方為實際值與預測值的時間序列對比圖，標註繁體中文『預測結果對比』，兩條曲線接近。中央下方為特徵重要性柱狀圖，柱子從高到低排列，標註繁體中文『特徵重要性 lag1>lag24>hour>rolling_mean>lag168』。柔和粉彩配色、白色背景、清晰的資訊圖表、infographic 風格。" --name 05_power_consumption_case_fig3_evaluation --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
