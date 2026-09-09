# 🎨 概念示意圖提示詞 — 模組7 特徵選擇與降維

> 對應講義：[`M7_Feature_Selection_and_Reduction.md`](../../modules/module_07_feature_selection/slides/M7_Feature_Selection_and_Reduction.md)
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：重要性是意見不是事實 / 四類方法的成本 / 選欄位也會洩漏
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；標籤要完全清晰再改 `--quality high`。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/slides_plan/M07_選擇與降維
```

---

### 圖 1 · 重要性是模型的意見
用在第 1 張。目的：先破除「重要性高就是好特徵」的直覺。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，中央一張特徵重要性長條圖，最長的那一條標示為『customerID』並打上紅色叉，旁邊一個人物皺眉說『這代表什麼』。第二長的『車齡』標綠色勾並附一句『折舊，講得出來』。圖表下方一行標語『排名高不等於有意義』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M7_fig1_importance_is_opinion --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 四類選擇方法的成本與風險
用在第 3 張。目的：用成本與是否綁模型兩個軸把四類方法排開。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，一個二維座標圖。橫軸標『計算成本，由低到高』，縱軸標『是否綁定特定模型，由否到是』。四個標籤放在對應位置：『過濾法』在左下、『PCA』在左上偏中、『嵌入法』在右中、『包裹法 RFE』在最右上並附紅色警告『很貴而且容易過度挑選』。每個標籤旁一句話說明何時用。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M7_fig2_selection_tradeoff --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 選欄位也會洩漏
用在第 7 張。目的：說明特徵選擇必須放進交叉驗證的每一折內。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，上下兩排。上排標紅色叉：『整份資料』→『選出前 20 個欄位』→『交叉驗證』，五個摺的圖示上都蓋著同一組已選欄位，標註『每一折都看過全部資料的答案』，分數 0.94。下排標綠色勾：『交叉驗證』外層先切五折，每一折內部各自執行『選欄位 → 訓練 → 評估』，標註『選擇也在 Pipeline 裡』，分數 0.87。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M7_fig3_selection_leak --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
