# 🎨 概念示意圖提示詞 — 標籤編碼與獨熱編碼

> 對應 notebook：`01_label_onehot_encoding.ipynb`（模組 M04 · 類別變數編碼）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：名目 vs 順序類別 / 標籤編碼對獨熱編碼 / 模型適用性對比
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_04_categorical_encoding/notebooks
```

---

### 圖 1 · 名目 vs 順序：類別的本質區別
目的：視覺化展示名目型與順序型類別的根本差異，幫助選擇適當編碼方法。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分左右兩側對比。左側代表名目型(Nominal)特徵：並排三個彩色圓形，分別標註『紅』『綠』『藍』，彼此之間無箭頭和順序。右側代表順序型(Ordinal)特徵：三個遞增的方塊，從小至大排列，標註『低』『中』『高』，中間有向上的箭頭表示遞增關係。柔和粉彩配色、白色背景、扁平infographic風格、繁體中文標籤清晰。" --name 01_label_onehot_fig1_nominal_ordinal --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 編碼方法對照：標籤編碼 vs 獨熱編碼
目的：並排展示兩種編碼方式的轉換過程，突出其原理與形狀差異。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，並排對比兩種編碼。左側為標籤編碼(Label Encoding)：一列原始類別『紅 綠 藍』轉換為『2 1 0』的單欄數字，標註『標籤編碼』『每個類別→單一整數』『不增加維度』。右側為獨熱編碼(One-Hot Encoding)：同樣的類別轉換為三欄二元矩陣『[1,0,0] [0,1,0] [0,0,1]』，標註『獨熱編碼』『每個類別→多欄0/1』『維度 = 類別數』。柔和粉彩、白色背景、乾淨infographic、繁體中文標籤清晰。" --name 01_label_onehot_fig2_encoding_methods --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 適用模型與風險提示
目的：呈現不同編碼方法對應的模型類型及使用陷阱。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分左右兩列展示編碼方法與模型的配對。左列標籤編碼(Label Encoding)上方接「樹模型(決策樹、隨機森林、XGBoost)」的綠色複選框，下方接「線性模型(邏輯迴歸、SVM)」的紅色禁止符。右列獨熱編碼(One-Hot Encoding)上方接「線性模型、距離模型(KNN)」的綠色複選框，下方接「高基數特徵」的黃色警告符標註『維度災難』。白色背景、柔和粉彩、infographic、繁體中文標籤清晰。" --name 01_label_onehot_fig3_model_risks --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
