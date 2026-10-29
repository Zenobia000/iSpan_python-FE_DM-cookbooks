# 🎨 概念示意圖提示詞 — 時間衍生特徵

> 對應 notebook：`03_time_derivatives.ipynb`（模組 M06 · 特徵創造）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：時間戳分解拆解 / 日曆與時間單位提取 / 時間差特徵計算
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_06_feature_creation/notebooks
```

---

### 圖 1 · 時間戳分解：從單一時間點到多維度組件
目的：展示如何將複雜的時間戳分解為年、月、日、時等獨立、可用的特徵。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，中央為一個大的時間戳『2023-10-27 14:30:45』，箭頭放射狀指向四周，每個方向分別分解為：頂部『年 2023、月 10、日 27』標註繁體中文『日期組件』；右側『時 14、分 30、秒 45』標註繁體中文『時間組件』；左側『星期五、第 300 天』標註繁體中文『週期資訊』；下方『是否週末、季度 Q4』標註繁體中文『派生特徵』。柔和粉彩配色、白色背景、放射狀分解圖示、繁體中文標籤清晰。" --name 03_time_derivatives_fig1_decomposition --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · pandas dt 訪問器提取時間特徵
目的：展示如何使用 dt 訪問器系統性地提取日曆和時間單位特徵。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，上方為原始時間序列列『timestamp』欄位，包含多個時間戳記；中間顯示 pandas dt 訪問器的各種方法調用，如『dt.year、dt.month、dt.day、dt.hour、dt.dayofweek』等，分別箭頭指向下方的輸出列；下方最終顯示一個豐富的特徵表格，各欄分別為年、月、日、時、星期幾等數值特徵。標註繁體中文『原始時間戳』『dt 訪問器』『日期組件』『時間組件』『星期資訊』『特徵豐富化』。柔和粉彩配色、白色背景、層級流程圖風格、繁體中文標籤清晰。" --name 03_time_derivatives_fig2_dt_accessor --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 時間差特徵：衡量事件間隔與累積時間
目的：展示如何計算時間間隔特徵，捕捉「流逝時間」的意義。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示時間軸上的多個事件點。上方時間軸顯示事件A（2023-10-01）、事件B（2023-10-15）、事件C（2023-10-27），下方標註各事件間的時間差特徵：『距離事件A的天數』分別標示為 0、14、26，以及『距離時間軸起點的累積天數』。用箭頭標示時間跨度。標註繁體中文『事件時間軸』『時間差計算』『距離起點14天』『距離起點26天』『時間流逝』『累積效應』。柔和粉彩配色、白色背景、時間軸圖示風格、繁體中文標籤清晰。" --name 03_time_derivatives_fig3_timediff --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
