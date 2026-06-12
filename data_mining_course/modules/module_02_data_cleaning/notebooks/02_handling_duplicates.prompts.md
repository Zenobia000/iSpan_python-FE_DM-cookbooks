# 🎨 概念示意圖提示詞 — 重複值處理

> 對應 notebook：`02_handling_duplicates.ipynb`（模組 M02 · 資料清理）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：重複值的危害 / 識別重複 / keep 參數對比
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_02_data_cleaning/notebooks
```

---

### 圖 1 · 重複值的危害
目的：清晰呈現重複資料對分析結果的扭曲影響。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分為左右對比。左側顯示一份含有重複行的資料表格（多個相同記錄），標註繁體中文『原始資料』『含重複值』；箭頻指向三個警告框，分別標註『扭曲統計』『模型偏見』『資料洩漏』；右側顯示同一個資料表格但已移除重複，標註繁體中文『清理後』『資料品質佳』；右側的計算結果更精確。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 02_handling_duplicates_fig1_impact --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 識別重複值：.duplicated() 方法
目的：展示布林遮罩如何標記重複行。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，左側顯示一個五行的資料表格，欄位為『actress』『movie』『rating』；第 0 行內容為『黃迪揚』『長安十二時辰』『8.5』；第 1、5 行內容相同；第 3、4 行只有『rating』不同。右側同位置顯示 .duplicated() 方法的結果，一列布林值『False』『True』『False』『False』『False』『True』，用紅色標記 True，用綠色標記 False。標註繁體中文『.duplicated()』『標記重複行』『第一次出現=False』『後續相同=True』。柔和粉彩配色、白色背景、教學圖表風格、繁體中文標籤清晰。" --name 02_handling_duplicates_fig2_identify --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · .drop_duplicates() 的 keep 參數對比
目的：直觀展示 keep='first' / 'last' / False 的差異。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，顯示三種並排的處理結果。頂部為原始資料表格，含有重複的行 A、A、A。下方分為三列，分別標題『keep=first』『keep=last』『keep=False』；第一列保留第一個 A；第二列保留最後一個 A；第三列全刪除。標註繁體中文『保留首次出現』『保留最後出現』『全部移除』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 02_handling_duplicates_fig3_keep_params --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
