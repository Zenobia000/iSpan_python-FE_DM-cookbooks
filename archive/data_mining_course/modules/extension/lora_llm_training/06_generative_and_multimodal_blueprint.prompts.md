# 🎨 概念示意圖提示詞 — 生成式與多模態大模型訓練藍圖

> 對應 notebook：`06_generative_and_multimodal_blueprint.ipynb`（模組 M11 · 大模型資料前處理與訓練）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：LLM 三階段訓練管線 / 文生圖 Diffusion 架構 / 多模態 VLM 融合
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_11_large_model_training/notebooks
```

---

### 圖 1 · LLM 三階段訓練管線

目的：從「預訓練」→「SFT 指令微調」→「偏好對齊」的全生命週期，展示資料與目標的演進。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示 LLM 訓練三階段。頂部三欄資料格式：『預訓練』純文字、『SFT』對話指令、『對齊』偏好資料。中間三個模型方塊演進：『基礎模型』→『指令模型』→『對齊模型』。底部三個目標：『學知識』→『學指令』→『學偏好』。柔和粉彩配色、白色背景、資訊圖表風格。" --name 06_generative_multimodal_fig1_llm_pipeline --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 文生圖 Diffusion 訓練架構

目的：展示「圖文配對資料 → VAE 潛在空間 + 文本編碼 → 去噪訓練」的過程。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示 Diffusion 訓練。頂部『圖文配對資料』並排圖像與文字。左支『圖像』→『VAE 編碼』→『潛在空間』；右支『文字』→『Text Encoder』→『文字編碼』；中央『去噪訓練』迭代步驟標註『條件：文字』；底部『微調技巧』LoRA、DreamBooth。柔和粉彩配色、白色背景、乾淨教學圖表。" --name 06_generative_multimodal_fig2_diffusion --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 多模態大模型 (VLM) 架構與融合

目的：展示視覺編碼器、文本編碼器與 LLM 解碼器如何融合成視覺語言模型。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示 VLM 多模態融合。頂部分兩路輸入：『影像』→『ViT 視覺編碼』→『視覺特徵』；『文字指令』→『Tokenize』→『文字嵌入』。中央『投影層』對齊特徵；匯合進入『LLM 解碼』；底部『生成輸出』看圖問答。柔和粉彩配色、白色背景、乾淨教學圖表。" --name 06_generative_multimodal_fig3_vlm_fusion --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
