# 概念示意圖提示詞：第 2 章　資料清理

> 對應講義：[`M2_Data_Cleaning_Fundamentals.md`](../../modules/module_02_data_cleaning/slides/M2_Data_Cleaning_Fundamentals.md)
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 2 張：清理在流程中的位置 / 去重必須在切分之前
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；標籤要完全清晰再改 `--quality high`。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/slides_plan/M02_資料清理
```

---

### 圖 1 · 清理的位置與順序
用在第 1 張。目的：說明清理排在哪裡，以及順序錯了會怎樣。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，一條由左至右的管線。管線依序為『原始檔案』『分塊讀取』『型態轉換』『去除重複』『文字清理』『切分訓練測試』『特徵工程』。在『去除重複』與『切分訓練測試』之間畫一條醒目的垂直虛線，標註『這條線之前做的事會影響兩邊』。管線入口放一個髒污的資料桶圖示，出口放一個乾淨的資料桶圖示。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M2_fig1_cleaning_order --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 重複值沒清乾淨會怎樣
用在第 3 張。目的：讓學員看見同一筆資料同時出現在訓練與測試的後果。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，左右對照。左側標紅色叉，一堆資料列中有兩列一模一樣並用相同顏色標示，切分後一列落在『訓練集』、另一列落在『測試集』，下方寫『模型等於考過同一題』，測試分數顯示 0.95。右側標綠色勾，先做去重讓重複列只剩一列，再切分，下方寫『測試題目模型沒看過』，分數顯示 0.81。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M2_fig2_duplicate_leak --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
