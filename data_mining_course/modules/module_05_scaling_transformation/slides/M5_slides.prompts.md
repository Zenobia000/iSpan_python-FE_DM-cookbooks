# 🎨 概念示意圖提示詞 — 模組5 特徵縮放與轉換

> 對應講義：`M5_Scaling_and_Transformation_Fundamentals.md`
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：誰需要縮放 / 尺度與分佈的差別 / 對樹縮放的代價
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；標籤要完全清晰再改 `--quality high`。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_05_scaling_transformation/slides
```

---

### 圖 1 · 哪些模型需要縮放
用在第 1 張。目的：先解決「這堂到底要不要做」的判斷。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，中央一條垂直分隔線。左側標題『需要縮放』，列出線性迴歸、邏輯迴歸、KNN、SVM、神經網路的小圖示，並畫一把尺與兩個距離不等的點說明『這些模型在算距離』。右側標題『不需要縮放』，列出決策樹、隨機森林、XGBoost 的小圖示，並畫一棵樹在某個閾值上分岔說明『樹只比大小，不算距離』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M5_fig1_who_needs_scaling --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 尺度與分佈是兩件事
用在第 3 張。目的：區分縮放與冪次轉換，這是本模組最常混淆的地方。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，上下兩排。上排標題『縮放：只改尺度』，左邊一個右偏的分佈曲線，箭頭指向右邊同樣右偏但橫軸數字變小的曲線，標註『形狀沒變』。下排標題『冪次轉換：改形狀』，左邊同一個右偏曲線，箭頭指向右邊接近對稱的鐘形曲線，標註『log 或 Yeo-Johnson』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M5_fig2_scale_vs_shape --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 對樹模型縮放是白做的
用在第 7 張。目的：用分數沒變但可解釋性下降來說明代價。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，左右對照。左側『原始欄位』，一棵決策樹的分岔條件寫著『里程 < 60000 公里』，旁邊一個人點頭表示看得懂，分數顯示 0.83。右側『縮放後』，同一棵樹的分岔條件變成『里程 < -0.42』，旁邊一個人皺眉打問號，分數同樣顯示 0.83，下方標註『分數一樣，可解釋性沒了』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M5_fig3_scaling_tree_waste --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
