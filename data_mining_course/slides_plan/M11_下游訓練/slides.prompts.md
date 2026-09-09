# 🎨 概念示意圖提示詞 — 模組十一 下游訓練

> 對應講義：[`M11_Large_Model_Training_Fundamentals.md`](../../modules/module_11_large_model_training/slides/M11_Large_Model_Training_Fundamentals.md)
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 4 張：同一批資料三種形狀 / 形狀對模型對任務 / 三種資料結構 / demo 不等於訓練大模型
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；標籤要完全清晰再改 `--quality high`。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/slides_plan/M11_下游訓練
```

---

### 圖 1 · 同一批資料，三種形狀三種任務
用在第 1 張。目的：把鐵律三講成一張圖。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，左側一堆相同的原始文件圖示，中央分出三條路徑。上路整成一排標示『input_ids 加標籤』的方塊，通向『分類器』與任務標籤『情感分類』。中路整成標示『對話 JSONL』的卡片，通向『SFT 微調』與任務標籤『指令跟隨』。下路整成一排標示『句向量』的點，通向『向量庫』與任務標籤『檢索問答』。標註『原始資料一樣，形狀不一樣，能做的事就不一樣』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M11_fig1_shape_decides --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 形狀、模型、任務對照
用在第 2 張。目的：這頁是大模型線的入口頁，要能單獨貼在牆上。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，一個三欄對照表。左欄標題『資料形狀』，由上而下列出『固定長度 token 序列』『對話格式 JSONL』『句向量』『影像張量』。中欄標題『接哪類模型』，對應列出『分類頭』『因果語言模型』『向量檢索』『視覺編碼器』。右欄標題『做什麼任務』，對應列出『分類與抽取』『指令跟隨』『問答與推薦』『影像分類』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M11_fig2_shape_model_task --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 三種資料結構要準備什麼
用在第 4 張。目的：讓學員知道每條路線的標籤格式與資料量門檻。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，三個並排卡片。左卡『分類』，示意一筆資料是一段文字加一個標籤，下方標註『幾百筆就能起步』。中卡『SFT』，示意一筆資料是使用者訊息加助理回覆的對話，下方標註『需要數千筆而且品質要一致』。右卡『檢索』，示意文件被切段後轉成向量存進資料庫，下方標註『不用訓練，重點在切段與去重』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M11_fig3_three_structures --size 1536x1024 --quality low --outdir concept_images
```

### 圖 4 · demo 不等於訓練大模型
用在第 8 張。目的：避免學員拿課堂 demo 的結果做過度宣稱。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，左右對照且比例懸殊。左側一個很小的方塊標註『課堂 demo：32 筆樣本、1 個 epoch、CPU、只證明流程跑得通』。右側一個大很多的方塊標註『真實微調：數萬筆、多輪、GPU、要有驗證集與基線比較』。兩者中間一個紅色驚嘆號標註『不要把左邊的結果講成右邊的能力』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M11_fig4_demo_not_training --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
