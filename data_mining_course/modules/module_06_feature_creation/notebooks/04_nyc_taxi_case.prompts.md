# 🎨 概念示意圖提示詞 — NYC 計程車案例實作

> 對應 notebook：`04_nyc_taxi_case.ipynb`（模組 M06 · 特徵創造）
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：NYC 計程車資料的特徵工程全景 / Haversine 距離計算 / 目標變數分佈轉換
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；若標籤需完全清晰可改 `--quality high`（成本較高）。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_06_feature_creation/notebooks
```

---

### 圖 1 · NYC 計程車特徵工程全景：從原始資料到豐富特徵
目的：展示綜合應用所有特徵工程技巧將原始計程車資料轉化為模型輸入。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，由左至右的轉化流程。左側為原始 NYC 計程車資料表，欄位包括『上車時間戳、下車時間戳、上車經度緯度、下車經度緯度、乘客數、行程時間（秒）』，標註繁體中文『原始資料』。箭頭指向中央，展示三層特徵創造：上層為時間衍生特徵（月、日、時、星期、是否週末），中層為地理交互特徵（Haversine 距離），下層為目標變數轉換（對數轉換）；右側為最終豐富的特徵表格。標註繁體中文『時間組件』『地理距離』『目標轉換』『可訓練的特徵集』。柔和粉彩配色、白色背景、多層次流程圖風格、繁體中文標籤清晰。" --name 04_nyc_taxi_case_fig1_overview --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · Haversine 距離：從經緯度座標計算球面距離
目的：直觀展示地理交互特徵的核心——球面距離計算。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，展示紐約地圖的簡化版本，標註上車點（A）和下車點（B）。A 點標示其經度緯度座標，B 點亦然。上車點和下車點之間用曲線連接，標註繁體中文『Haversine 距離』『地球球面』。下方展示 Haversine 公式的概念（無需完整數學式），用箭頭指向最終的距離值『距離 3.5 公里』。標註繁體中文『上車點座標』『下車點座標』『球面距離計算』『行程距離』『關鍵預測特徵』。柔和粉彩配色（藍綠為主表地圖）、白色背景、地理圖示風格、繁體中文標籤清晰。" --name 04_nyc_taxi_case_fig2_haversine --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 目標變數分佈轉換：右偏到常態
目的：展示為何對行程時間進行對數轉換，改善模型訓練。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，分為上下兩行對比。上行左側顯示『原始行程時間』的分佈直方圖，呈現明顯右偏（長尾拖向右）的形狀，標註繁體中文『右偏分佈』『長尾』；上行右側顯示其統計資訊『偏度高、異常值多、模型不適』。下行左側顯示『對數轉換後 log(1+duration)』的分佈直方圖，更接近常態分佈（鐘形），標註繁體中文『接近常態分佈』『更對稱』；下行右側顯示改善後的統計資訊『偏度降低、分佈正常化、模型友善』。中間有箭頭指示轉換過程。柔和粉彩配色、白色背景、對比直方圖風格、繁體中文標籤清晰。" --name 04_nyc_taxi_case_fig3_distribution_transform --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
