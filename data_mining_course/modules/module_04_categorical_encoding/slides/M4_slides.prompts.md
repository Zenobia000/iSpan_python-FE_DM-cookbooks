# 🎨 概念示意圖提示詞 — 模組4 類別變數編碼

> 對應講義：`M4_Categorical_Encoding_Handout.md`
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：編碼是在學對照表 / 四種編碼的取捨 / Target encoding 的洩漏
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；標籤要完全清晰再改 `--quality high`。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_04_categorical_encoding/slides
```

---

### 圖 1 · 編碼在學一張對照表
用在第 1 張。目的：把「編碼會 fit 出參數」這件事視覺化。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，中央一個標示『編碼器』的方塊，左側輸入是一欄文字類別『台北 台中 高雄』，右側輸出是數字欄。方塊內部畫出一張小小的對照表把文字對到數字，並用醒目框標註『這張表就是參數，`fit` 學到的東西』。方塊下方一條箭頭指向『只能從訓練集學』的標籤。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M4_fig1_encoding_is_a_table --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 四種編碼的取捨
用在第 3 張。目的：一張圖看完四種編碼的代價與風險。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，四欄比較圖。四欄依序為『Label 標籤』『One-hot 獨熱』『Count 計數』『Target 目標』。每欄用三個小圖示標示：產生幾欄（方塊數量）、會不會引入假的順序（一把尺）、洩漏風險（放大鏡）。Label 的尺標紅色警告，One-hot 的方塊畫成一長排標註『高基數會爆維度』，Target 的放大鏡標最大的紅色警告並註明『需要 out-of-fold』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M4_fig2_encoding_tradeoff --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · Target encoding 洩漏的樣子
用在第 7 張。目的：用訓練與測試分數的落差說明洩漏。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，左右對照。左側標紅色叉，一張資料表中某一列的目標值用箭頭繞回去參與計算自己的編碼值，形成一個迴圈，下方兩個長條顯示『訓練 AUC 0.97』與『測試 AUC 0.61』，落差用紅色標出。右側標綠色勾，資料被切成五個摺，計算某一摺的編碼時其他四摺提供數值、該摺自己被排除，下方兩個長條顯示『訓練 0.86』與『測試 0.84』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M4_fig3_target_leak --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
