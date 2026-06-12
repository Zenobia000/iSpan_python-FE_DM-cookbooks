# 🎨 概念示意圖提示詞 — 日期與時間特徵 (Date and Time Features)

> 對應 notebook：`03_date_time_features.ipynb`（模組 M08 · 時間序列特徵工程）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：時間成分提取 / 週期性特徵與模式 / 正弦餘弦編碼
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_08_time_series/notebooks
```

---

### 圖 1 · 時間戳分解：從日期時間提取多維特徵
目的：展示如何從原始時間戳中提取年、月、日、小時等多維度特徵。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示時間戳分解過程。中央為一個日期時間字符串『2024-03-15 14:30:00 週五』，周圍八個箭頭分別指向八個特徵方框，分別標註繁體中文『年份 2024』『月份 03』『日期 15』『小時 14』『分鐘 30』『星期幾 4』『季度 Q1』『一年中第幾天』。方框配色柔和粉彩，整體佈局呈放射狀。白色背景、清晰的資訊圖表、infographic 風格。" --name 03_date_time_features_fig1_extraction --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 週期性模式與目標值的關係
目的：展示日期時間特徵如何捕捉週期性模式（如季節性、星期幾效應）。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示三個並排的箱形圖。左側箱形圖標註繁體中文『按月份 季節性：冬夏高耗電』，展示 12 個月份的箱體；中間箱形圖標註繁體中文『按星期幾 週末效應：工作日vs休息日』，展示 7 個箱體；右側箱形圖標註繁體中文『按小時 日內週期：峰谷模式』，展示 24 個小時的箱體。箱體顏色柔和粉彩，縱軸為目標值，清晰展示週期性規律。白色背景、infographic 風格。" --name 03_date_time_features_fig2_periodicity --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 週期性特徵編碼：正弦/餘弦變換
目的：展示如何將環形特徵（月份、小時）轉換為兩個維度避免模型誤解順序。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示週期性編碼的概念。上方左側為一個圓形鐘面，標註 0-11 月份均勻分佈在圓周，標註繁體中文『月份環形性：12月與1月相近』。上方右側展示兩個正弦曲線圖分別標註繁體中文『月份正弦編碼 sin(2π*month/12)』『月份餘弦編碼 cos(2π*month/12)』，曲線平滑循環。下方展示一個 3 欄的表格：『月份(原始)』『正弦特徵』『餘弦特徵』，含數值示例。柔和粉彩配色、白色背景、清晰的資訊圖表、繁體中文標籤。" --name 03_date_time_features_fig3_sincos --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
