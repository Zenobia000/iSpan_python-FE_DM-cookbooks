# 🎨 概念示意圖提示詞 — Subword Tokenization

> 對應 notebook：`02_tokenization.ipynb`（模組 M09 · 多模態特徵（文字））
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：文本切分粒度對比 / Tokenizer 處理管線 / Subword 詞彙表構建過程
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_09_multimodal_features/notebooks/01_text_features
```

---

### 圖 1 · 三種文本切分粒度：詞 vs 字元 vs Subword

目的：視覺比較為何 subword 是現代模型的最優折衷。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，並排展示三種切分粒度。左列標示『詞級』，完整詞語、詞表龐大；中列標示『字元級』，逐字切分、序列極長；右列標示『Subword』，用綠色背景突出最佳選擇、平衡詞表與序列。每列展示實際切分效果。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 02_tokenization_fig1_granularity --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · Tokenizer 處理管線：從文本到整數張量

目的：展示文本經過 tokenizer 後的完整轉換流程與輸出形狀。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右的水平流程。第一格為原始文本『I love NLP』；第二格標示『切分』；第三格標示『Token ID』，顯示整數序列；第四格標示『張量』，展示 input_ids 與 attention_mask。流程用箭頭串接。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 02_tokenization_fig2_pipeline --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · BPE 詞彙表的遞進構建：從字元到 Subword

目的：直觀展示 BPE 演算法如何逐步合併高頻字元對形成詞彙表。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由上至下展示 BPE 遞進過程。最上層標示『初始：字元』，單一字元列表；中間層標示『迭代1、迭代2』，逐步合併高頻對象；最下層標示『詞彙表』。每層用不同顏色突出新增單位，箭頭指向下層。柔和粉彩配色(藍/綠/橙)、白色背景、infographic 風格、繁體中文標籤清晰。" --name 02_tokenization_fig3_bpe_vocab --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
