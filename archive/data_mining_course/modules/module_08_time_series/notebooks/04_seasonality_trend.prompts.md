# 🎨 概念示意圖提示詞 — 季節性與趨勢分解 (Seasonality and Trend Decomposition)

> 對應 notebook：`04_seasonality_trend.ipynb`（模組 M08 · 時間序列特徵工程）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：時間序列分解的三成分 / 加法模型vs乘法模型 / seasonal_decompose 結果視覺化
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_08_time_series/notebooks
```

---

### 圖 1 · 時間序列分解的三大成分
目的：展示原始時間序列如何分解為趨勢、季節性與殘差三個獨立成分。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由上至下堆疊四個時間序列圖。最上方標註繁體中文『原始時間序列』，顯示高低起伏的波動曲線。下方依次為三個分解成分：第一個標註繁體中文『趨勢成分 Trend：長期變化方向』，顯示平滑的上升曲線；第二個標註繁體中文『季節性成分 Seasonality：重複週期波動』，顯示規律的鋸齒波形；第三個標註繁體中文『殘差成分 Residual：隨機波動與異常』，顯示分散的零均值波動。四個圖對齊橫軸，清晰展示分解關係。柔和粉彩配色、白色背景、infographic 風格。" --name 04_seasonality_trend_fig1_decomposition --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 加法模型 vs 乘法模型
目的：對比兩種分解模型的適用場景與數學關係。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示兩個並排的模型框架。左側標註繁體中文『加法模型 Y(t) = 趨勢 + 季節性 + 殘差 適用於：變動幅度恆定』，下方展示一個時間序列圖，季節性波動的高度保持一致。右側標註繁體中文『乘法模型 Y(t) = 趨勢 × 季節性 × 殘差 適用於：變動幅度隨趨勢增加』，下方展示一個時間序列圖，季節性波動的高度逐漸擴大。兩側對比清晰，柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤。" --name 04_seasonality_trend_fig2_additive_vs_multiplicative --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · seasonal_decompose 函數與應用
目的：展示如何使用統計方法進行時間序列分解，以及提取的成分如何用作特徵。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示完整的分解流程。左側為一個標註繁體中文『原始時間序列』的波動曲線。中央顯示代碼片段『from statsmodels import seasonal_decompose decomposition = seasonal_decompose(ts, model=additive, period=12)』。右側展示分解後四個成分的堆疊視圖：『原始』『趨勢』『季節性』『殘差』四個小面板。下方補充文本框標註繁體中文『應用：將趨勢與季節成分作為模型特徵、去季節化、去趨勢、異常檢測』。柔和粉彩配色、白色背景、清晰的資訊圖表。" --name 04_seasonality_trend_fig3_decompose_method --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
