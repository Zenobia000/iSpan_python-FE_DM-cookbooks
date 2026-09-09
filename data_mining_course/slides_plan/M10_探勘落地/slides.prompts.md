# 概念示意圖提示詞：第 10 章　資料探勘應用

> 對應講義：[`M10_Data_Mining_Applications_Fundamentals.md`](../../modules/module_10_data_mining_applications/slides/M10_Data_Mining_Applications_Fundamentals.md)
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：工具接成一條線 / 三種落地方式 / ID 變成最重要特徵
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；標籤要完全清晰再改 `--quality high`。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/slides_plan/M10_探勘落地
```

---

### 圖 1 · 工具接成一條線
用在第 1 張。目的：讓學員看見前面八個模組在這裡合流。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，一條由左至右的管線，入口是原始資料表。管線上依序掛著標籤『清理』『缺值』『編碼』『縮放』『特徵創造』『特徵選擇』，這些全部被一個大框框起來標註『Pipeline 內，切分之後才 fit』。管線出口分成三個方向，分別是『樹模型預測』『分群』『關聯規則』三個小圖示。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M10_fig1_end_to_end --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 三種落地方式的選用
用在第 3 張。目的：用問題型態決定方法，取代逐一講演算法步驟。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，一個三分支的決策樹。根節點寫『你要回答什麼』。分支一『這筆會不會發生』指向樹模型的圖示，標註『XGBoost、LightGBM，可以看重要性』。分支二『這些對象可以分成幾群』指向散點分群的圖示，標註『K-Means 要先縮放、DBSCAN 對參數敏感』。分支三『哪些東西常一起出現』指向購物籃圖示，標註『關聯規則，只帶一小段』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M10_fig2_three_applications --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · ID 變成最重要的特徵
用在第 7 張。目的：這是落地階段最常見、也最容易被忽略的錯。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，左側一張特徵重要性長條圖，最長的一條是『customerID_encoded』並打上紅色叉。右側一個人物指著它問『流水號為什麼會影響流失』，下方畫出真相：一排客戶編號依照建檔時間遞增，箭頭指向『編號其實在代表年資』，標註『模型記住的是資料建檔順序』。最下方一行結論『對不回理由就先移除，再看分數掉多少』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M10_fig3_id_importance --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
