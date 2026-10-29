# 🎨 概念示意圖提示詞 — 資料視覺化基礎

> 對應 notebook：`02_data_visualization.ipynb`（模組 M01 · 探索性資料分析入門）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：單變數分佈圖鑑 / 雙變數關係 / 選圖決策
>
> ⚠️ 含中文字的圖預設 `--quality low`；標籤需完全清晰可改 `--quality high`。
> ▶️ 執行前先 `cd` 到本資料夾，圖輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_01_eda_intro/notebooks
```

---

### 圖 1 · 單變數分佈圖鑑
目的：對照看懂直方圖、KDE、箱型圖、計數圖各自表達什麼。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，2x2 四宮格，每格一個簡化的統計圖示與繁體中文標題：左上『直方圖 Histogram』、右上『核密度 KDE 曲線』、左下『箱型圖 Boxplot』、右下『計數圖 Countplot』。最上方一條總標題帶『單變數分佈：看一個變數長什麼樣子』。柔和粉彩配色、白色背景、乾淨 infographic 風格、繁體中文標籤清晰。" --name 02_data_visualization_fig1_univariate --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 雙變數關係：散點圖
目的：用散點圖呈現兩個數值變數之間的關係與趨勢。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，一張帶 x 軸與 y 軸的散點圖，許多資料點呈現正相關走勢，並畫一條穿過點群的趨勢線。標註繁體中文『散點圖：看兩個變數的關係』『趨勢線』與座標軸『變數 X』『變數 Y』。柔和粉彩配色、白色背景、infographic 風格、繁體中文標籤清晰。" --name 02_data_visualization_fig2_scatter --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 該選哪種圖？決策示意
目的：依變數型態（數值／類別、單一／成對）快速選對圖。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，一個簡單的選圖決策樹。頂端問題卡標註繁體中文『我要看什麼？』，分出兩條路徑：『單一變數的分佈』指向直方圖/箱型圖小圖示；『兩個變數的關係』再分為『數值對數值→散點圖』與『類別對數值→箱型圖/長條圖』。以箭頭連接、各節點皆有繁體中文標籤。柔和粉彩配色、白色背景、乾淨教學圖表。" --name 02_data_visualization_fig3_choose_plot --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
