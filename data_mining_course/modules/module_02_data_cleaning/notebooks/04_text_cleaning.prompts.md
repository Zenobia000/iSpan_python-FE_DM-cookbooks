# 🎨 概念示意圖提示詞 — 文字欄位清理

> 對應 notebook：`04_text_cleaning.ipynb`（模組 M02 · 資料清理）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：文字噪音類型 / 向量化字串操作 / 鏈式清理流程
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_02_data_cleaning/notebooks
```

---

### 圖 1 · 文字中常見的噪音類型
目的：展示四類主要的文字噪音及其來源。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，四個並排的框框，各展示一種文字噪音。第一框『大小寫不一致』，顯示『Apple』『apple』兩個不同的詞；第二框『多餘空白』，顯示『 a dog』『a dog 』『a dog』三個不同的形式；第三框『標點符號』，顯示『Hello!』『Hello』被視為不同；第四框『特殊字元』，顯示『price$99』『price99』的差異。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 04_text_cleaning_fig1_noise_types --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · .str 向量化字串操作
目的：展示 .str 方法如何批量處理 Series 中的所有字串。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，左側顯示一個 Pandas Series，含有五個髒亂的字串元素『 Apple 』『BANANA！』『  orange  』『CaRrOt!!!』『  GRAPE  』；箭頻指向中央一個『.str 向量化操作』的框，內含並排的三個方法『.lower()』『.strip()』『.replace()』；右側顯示清理後的結果『apple』『banana』『orange』『carrot』『grape』。柔和粉彩配色、白色背景、教學流程圖、繁體中文標籤清晰。" --name 04_text_cleaning_fig2_str_methods --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 鏈式清理操作的全流程
目的：展示 .lower() → .strip() → .replace() 依序應用的效果。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由上至下的流程。頂部顯示原始髒資料『 " HELLO, WORLD! " 』；箭頻指向第一步『.lower()』，結果『 " hello, world! " 』；再箭頻指向第二步『.strip()』，移除前後空白，結果『" hello, world! "』；最後箭頻指向第三步『.replace(r\'[^\w\s]\', \'\')』移除標點符號，最終清理結果『hello world』。柔和粉彩配色、白色背景、step-by-step 流程圖風格、繁體中文標籤清晰。" --name 04_text_cleaning_fig3_chaining --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
