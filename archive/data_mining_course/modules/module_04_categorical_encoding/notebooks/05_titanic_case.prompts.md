# 🎨 概念示意圖提示詞 — 案例實戰 Titanic 資料集

> 對應 notebook：`05_titanic_case.ipynb`（模組 M04 · 類別變數編碼）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：Titanic 資料集概覽 / 策略決策流程 / 編碼轉換全景
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_04_categorical_encoding/notebooks
```

---

### 圖 1 · Titanic 資料集與類別特徵識別
目的：介紹 Titanic 資料集中的主要特徵及其類別屬性，建立案例背景。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示鐵達尼號沉船的背景圖示(簡化)，下方為一個表格摘錄，展示 Titanic 資料集的幾個關鍵特徵：『Sex』(男/女)、『Embarked』(S/C/Q三港口)、『Age』(數值)、『Pclass』(1/2/3艙等級)、『Survived』(0/1存活)。用色塊標註『數值特徵』(藍色)與『類別特徵』(黃色)，標題『Titanic資料集』『目標：預測存活』。柔和粉彩、白色背景、扁平infographic、繁體中文標籤清晰。" --name 05_titanic_fig1_dataset_overview --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 編碼策略決策樹
目的：展示如何根據特徵的類型與基數選擇最恰當的編碼方法。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，一個決策流程圖(Decision Tree)：起點『評估類別特徵』分為『是否順序型？』。若『是』→『標籤編碼』；若『否』→『基數多少？』分為『低(≤10)』→『獨熱編碼』、『高(>10)』→『特徵哈希或合併』。右側為Titanic的具體應用：『Sex(二元)』→『標籤編碼(0/1)』、『Embarked(3值)』→『獨熱編碼』、『Age(數值，已有)』→『保持原樣或分箱創序序特徵』。柔和粉彩、白色背景、infographic、繁體中文標籤清晰。" --name 05_titanic_fig2_strategy_decision --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 完整編碼轉換全景
目的：展示原始 Titanic 資料如何通過多種編碼方法轉換為機器學習就緒的數值化表格。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由上至下展示三層：頂層『原始資料』展示一個小表格含『Sex(Male/Female)』『Embarked(S/C/Q)』『Age(數值)』『Survived(0/1)』；中層『編碼過程』展示三種轉換：『Sex→標籤編碼(0/1)』、『Embarked→獨熱編碼(三欄)』、『Age→分箱創序序特徵(低/中/高)』；底層『最終數值化表格』展示完全數值化的DataFrame，所有欄位都是數字，標註『完全數值化、機器學習就緒』。柔和粉彩、白色背景、infographic、繁體中文標籤清晰。" --name 05_titanic_fig3_full_pipeline --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
