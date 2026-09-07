# 🎨 概念示意圖提示詞 — 文字下游微調與檢索

> 對應 notebook：`02_text_downstream.ipynb`（模組 M11 · 大模型資料前處理與訓練）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：三條文字下游路線 / 分類微調流程 / RAG 檢索管線
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_11_large_model_training/notebooks
```

---

### 圖 1 · 文字下游三條路線

目的：展示文字資料可以走的三條路：分類微調、LLM 指令微調、檢索增強生成。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，一個分叉路線圖。起點為『文字資料』，分出三條路：路線 A『分類微調』→『DistilBERT』、路線 B『LLM 微調』→『LLM+LoRA』、路線 C『RAG 檢索』→『向量檢索』。各路終點標註對應任務簡稱（『分類』『對話』『問答』）。柔和粉彩配色、白色背景、乾淨資訊圖表。" --name 02_text_downstream_fig1_three_routes --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 分類微調完整流程

目的：串起「讀資料 → 前處理 → 訓練 → 評估」的完整文本分類微調循環。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由上至下的流程圖。第一格『文本+標籤』資料→箭頭『Tokenize』→『input_ids』；第二格『DistilBERT』模型；箭頭『Trainer 訓練』→『Loss 曲線』；第三格『評估指標』（準確率、F1）。柔和粉彩配色、白色背景、資訊圖表風格、標籤簡潔。" --name 02_text_downstream_fig2_classification_flow --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · RAG 檢索增強生成架構

目的：展示「檢索」與「生成」兩步的協作：用句向量找相關文件，再塞進 LLM prompt 生成答案。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示 RAG 流程。左側『知識庫』多文件堆疊；中上『句向量編碼器』轉化為向量；右側『查詢輸入』；箭頭『語意相似度』→『檢索前K文件』；指向下方『LLM』標註『生成答案』；最下『輸出』。柔和粉彩配色、白色背景、乾淨教學圖表。" --name 02_text_downstream_fig3_rag_pipeline --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
