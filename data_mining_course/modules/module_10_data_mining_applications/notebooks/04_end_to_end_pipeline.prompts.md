# 🎨 概念示意圖提示詞 — 端到端資料探勘流程

> 對應 notebook：`04_end_to_end_pipeline.ipynb`（模組 M10 · 資料探勘應用）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：資料探勘完整流程 / Pipeline 與 ColumnTransformer 架構 / 迭代優化循環
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_10_data_mining_applications/notebooks
```

---

### 圖 1 · 資料探勘六大階段完整流程

目的：展示從原始數據到模型部署的六個核心階段，理解端到端流程的全貌。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，六個並排的流程步驟。『定義』『收集』『分析』『工程』『訓練』『部署』。各步驟用箭頭連接。簡潔標籤。柔和粉彩配色、白色背景、infographic 風格。" --name 04_end_to_end_pipeline_fig1_stages --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · Pipeline 與 ColumnTransformer 架構

目的：展示 scikit-learn 中 Pipeline 與 ColumnTransformer 如何組織預處理與模型訓練的自動化流程。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示層級化處理結構。頂層『ColumnTransformer』分支三路：『數值』『類別』『日期』；各路分別處理後合併進『模型』。簡潔標籤。柔和粉彩配色、白色背景、infographic 風格。" --name 04_end_to_end_pipeline_fig2_architecture --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 迭代優化與持續改進

目的：展示資料探勘的迭代性，不斷調整特徵、模型、超參數以達成更好的性能。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示循環迭代過程。中心『模型評估』，四個外圍箭頭：『特徵』『演算』『參數』『資料』，各方向返回迴圈。標籤『改進』『優化』『迭代』。柔和粉彩配色、白色背景、infographic 風格。" --name 04_end_to_end_pipeline_fig3_iteration --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
