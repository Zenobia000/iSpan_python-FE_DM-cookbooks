# 🎨 概念示意圖提示詞 — 模組6 特徵創造

> 對應講義：`M6_Feature_Creation_Fundamentals.md`
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：從因果鏈長出特徵 / 三類創造方式 / 聚合特徵的洩漏
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；標籤要完全清晰再改 `--quality high`。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/modules/module_06_feature_creation/slides
```

---

### 圖 1 · 從因果鏈長出特徵
用在第 1 張。目的：說明特徵是從因果鏈長出來的，不是欄位相乘湊出來的。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，左右對照。左側標紅色叉，一台機器把一堆欄位兩兩相乘噴出大量標示問號的新欄位，標註『AI 生成 80 個交叉項，講不出理由』。右側標綠色勾，一條由上而下的因果鏈『折舊沒走完 → 車齡小殘值高 → 價格高』，從中間那一層拉出一個欄位卡片『car_age = 快照年 − 出廠年』，標註『一句話講得完』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M6_fig1_from_cause_to_feature --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 三類創造方式
用在第 3 張。目的：把交互、聚合、日期拆欄三類的用途與風險擺在一起。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，三個並排卡片。左卡『交互項』，兩個欄位相乘變成一欄，標註『用在兩個因素要一起看的時候』。中卡『分組聚合』，一堆列依品牌分組後算出群平均，標註紅色警告『只能用訓練集算』。右卡『日期拆欄』，一個日期被拆成年、月、星期幾、是否假日四欄，標註『把時間變成模型看得懂的欄位』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M6_fig2_three_kinds --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 聚合特徵的洩漏
用在第 7 張。目的：這是本模組最常見也最難察覺的錯誤。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，上下兩排流程對照。上排標紅色叉：『整份資料』→『依品牌算平均價』→『把群平均併回每一列』→『隨機切分』，在切分處畫一個標示『測試集的價格已經混進訓練特徵裡』的紅色警示牌。下排標綠色勾：『整份資料』→『先切分』→『只用訓練集算品牌平均價』→『併回訓練集與測試集』，標註綠色『測試集只是查表，沒有貢獻數值』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M6_fig3_aggregation_leak --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
