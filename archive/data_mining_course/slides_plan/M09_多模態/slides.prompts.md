# 概念示意圖提示詞：第 9 章　多模態特徵工程

> 對應講義：[`M9_Multimodal_Features_Fundamentals.md`](../../modules/module_09_multimodal_features/slides/M9_Multimodal_Features_Fundamentals.md)
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 4 張：換模態不換腦袋 / 四模態總覽 / 取樣策略的取捨 / 資料污染
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；標籤要完全清晰再改 `--quality high`。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/slides_plan/M09_多模態
```

---

### 圖 1 · 換模態不換腦袋
用在第 1 張。目的：先講清楚四種模態共用同一套處理骨架。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，四條並排的水平流程，左端分別是文字文件、照片、聲波、影片膠捲四個圖示。四條流程中間都經過同樣的三個階段方塊『理解』『清理』『導入張量』，右端都收斂到同一個標示『張量』的立方體。整體用一個大括號把四條流程括起來，標註『骨架相同，只有前處理工具不同』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M9_fig1_same_skeleton --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 四模態總覽
用在第 2 張。目的：一頁看完四種模態的原始形式、目標張量與常用模型。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，一個四列三欄的表格式圖解。左欄為模態圖示：文字、影像、音訊、影片。中欄為目標張量形狀，分別標註『input_ids (B, L)』『pixel_values (B, 3, H, W)』『waveform (B, T)』『frames (B, T, 3, H, W)』。右欄為常用模型名稱：BERT 類、ViT 與 CLIP、Whisper 與 wav2vec2、VideoMAE。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M9_fig2_four_modalities --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 取樣策略的取捨
用在第 5 張。目的：音訊取樣率與影片影格取樣是這堂最容易做錯的地方。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，上下兩排。上排『音訊取樣率』，同一段聲波分別以高取樣率與低取樣率呈現，高的標註『資訊完整但檔案大』，低的標註『省空間但高頻細節掉了』，中間一個警告標籤寫『整批必須統一』。下排『影片影格取樣』，一條影片時間軸上三種取法：均勻取樣、只取開頭、隨機取樣，各附一句適用情境與風險。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M9_fig3_sampling_tradeoff --size 1536x1024 --quality low --outdir concept_images
```

### 圖 4 · 資料污染
用在第 8 張。目的：說明多模態與大模型情境下洩漏換了一種形式。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，左右對照。左側標紅色叉，一大桶『訓練語料』裡混進幾張標示『測試題目』的卡片，模型圖示旁的分數顯示 0.96，下方一個上線後的畫面顯示分數掉到 0.62，標註『不是模型不好，是它背過答案』。右側標綠色勾，訓練語料在進入模型前先經過一個標示『去重與去污染』的濾網，濾出測試題目卡片，分數顯示 0.81 並標註『這個數字才拿得出去』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M9_fig4_contamination --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
