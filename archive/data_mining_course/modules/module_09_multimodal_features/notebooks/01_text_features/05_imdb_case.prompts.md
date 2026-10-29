# 🎨 概念示意圖提示詞 — IMDB 情感分析案例

> 對應 notebook：`05_imdb_case.ipynb`（模組 M09 · 多模態特徵（文字））
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：經典 vs 現代兩條路線並行 / 特徵表示對比 / 模型性能與維度差異
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/01_text_features
```

---

### 圖 1 · 經典 vs 現代兩條文本分類管線

目的：並行展示 TF-IDF 路線與句向量路線的完整流程。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，兩條並列的管線流程。上方『路線 A』標示『TF-IDF + 邏輯回歸』，流程：文本 → TF-IDF → 訓練 → 預測。下方『路線 B』標示『句向量 + 邏輯回歸』，流程：文本 → 句向量 → 訓練 → 預測。兩條線起終點對齊便於對比。用橙色與藍色區隔。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 05_imdb_case_fig1_pipelines --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 稀疏 TF-IDF vs 稠密句向量的特徵空間對比

目的：展示兩種特徵表示在維度與密度上的根本差異。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分左右兩個特徵空間。左側標示『TF-IDF 稀疏』，展示寬矩陣，大量白色零值，密度極低；標註『高維、稀疏、可解釋』。右側標示『句向量稠密』，展示低維空間，正評綠點與負評紅點形成清晰分群；標註『低維、稠密、聚集』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 05_imdb_case_fig2_features --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 性能對比：準確度與維度權衡

目的：量化展示兩條路線的效能與效率差異。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，比較圖表展示兩路線指標。條形圖顯示準確度（TF-IDF 約 88%、句向量約 92%）與維度（TF-IDF 約 5000、句向量約 384）。下方標示『遷移性』，TF-IDF 為『重訓』，句向量為『零次學習』。表格下標註『現代方案更優』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 05_imdb_case_fig3_performance --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
