# 🎨 概念示意圖提示詞 — 目標編碼

> 對應 notebook：`03_target_encoding.ipynb`（模組 M04 · 類別變數編碼）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：目標編碼原理 / 資料洩漏陷阱 / 交叉驗證防護與平滑技術
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_04_categorical_encoding/notebooks
```

---

### 圖 1 · 目標編碼的核心原理
目的：展示如何用類別對應的目標均值直接進行編碼。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示三個城市『紐約』『舊金山』『芝加哥』各自對應的購買記錄。紐約：3筆資料中2筆購買(1,1,0)→均值0.67；舊金山：2筆資料中1筆購買(1,0)→均值0.5；芝加哥：2筆資料中1筆購買(1,0)→均值0.5。下方展示轉換：城市名稱被替換為對應的目標均值，標註『目標編碼(Mean Encoding)』『用目標均值直接編碼』『強力的預測信號』。柔和粉彩、白色背景、扁平infographic、繁體中文標籤清晰。" --name 03_target_encoding_fig1_principle --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 天真方法的資料洩漏陷阱
目的：警示直接使用全資料計算均值會造成過擬合。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分左右對比。左側標題『危險：天真方法』，展示一個資料表格，紅色箭頭從第一列（『紐約 購買=1』）指向計算過程『計算紐約均值時包含了第一列自身的購買值』，標註『資料洩漏』『模型在訓練時作弊』。右側標題『後果：過度擬合』，展示訓練精度高(綠色)但測試精度低(紅色)的對比圖，警告符號突出『在測試集表現遠低於訓練集』。柔和粉彩、白色背景、infographic、繁體中文標籤清晰。" --name 03_target_encoding_fig2_data_leakage --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 交叉驗證防護 & 平滑技術
目的：展示如何用 K-Fold 交叉驗證和平滑來增強穩健性。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，上半部為交叉驗證(K-Fold)：展示5折分割，第1-4折(淺藍)用於計算編碼，第5折(亮藍)為驗證折，標註『K-Fold交叉驗證』『計算時排除該列本身』『防止資料洩漏』。下半部為平滑技術(Smoothing)：展示低頻類別(只出現1-2次)的編碼值經過加權平均『w×局部均值 + (1-w)×全域均值』，從不穩定的單值變為更可靠的平滑值，標註『平滑技術』『信任度 = 樣本數 / (樣本數 + m)』『低頻類別更穩定』。柔和粉彩、白色背景、infographic、繁體中文標籤清晰。" --name 03_target_encoding_fig3_cv_smoothing --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
