# 🎨 概念示意圖提示詞 — Instacart 購物籃分析

> 對應 notebook：`02_instacart_case.ipynb`（模組 M10 · 資料探勘應用）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：Instacart 資料特徵 / 購物籃轉換流程 / 購買模式洞察
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_10_data_mining_applications/notebooks/01_association_rules
```

---

### 圖 1 · Instacart 資料集結構概覽

目的：展示 Instacart 大型交易資料集的多個表格與關鍵欄位，理解真實零售數據的結構。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示三張互相連接的資料表。左側『訂單表』；中間『訂單商品表』；右側『商品表』。用箭頭連接顯示關聯。柔和粉彩配色、白色背景、資訊圖表(infographic)風格。" --name 02_instacart_case_fig1_schema --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 購物籃轉換為交易矩陣

目的：展示原始購物籃資料經過 One-Hot 編碼轉換為適合關聯規則探勘的二元矩陣的全過程。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右的轉換流程。左側為原始購物籃；中間為『One-Hot編碼』箭頭；右側為二元矩陣（0或1）。柔和粉彩配色、白色背景、infographic 風格。" --name 02_instacart_case_fig2_encoding --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 實際購買模式與商業洞察

目的：展示從 Instacart 真實數據中挖掘出的高置信度購買模式及其商業應用價值。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示三條關聯規則流向。左『蔬菜→牛奶』；中『麵包→果汁』；右『起司→葡萄酒』。右側應用方塊：『推薦』『促銷』『配置』。柔和粉彩配色、白色背景、infographic 風格。" --name 02_instacart_case_fig3_patterns --size 1536x1024 --quality low --outdir concept_images
```

---

> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
