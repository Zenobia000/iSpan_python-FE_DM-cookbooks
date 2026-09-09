# 🎨 概念示意圖提示詞 — 總整 英國二手車

> 對應講義：`C5_二手車總整.md`
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 2 張：二手車的 SCQA / 聚合特徵不洩漏的做法
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；標籤要完全清晰再改 `--quality high`。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/capstone/slides
```

---

### 圖 1 · 二手車的 SCQA
用在第 1 張。目的：把問題定義用一張圖固定下來，後面每一步都回頭對這張。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，一個四格方陣。左上『情境』畫一個二手車經銷商站在一排車前面，配文字『每天要為上架車輛開價』。右上『衝突』畫一個天平，一端標『開太高賣不掉』另一端標『開太低沒毛利』。左下『問題』畫一個問號與價格標籤，配文字『哪些因素在驅動價格』。右下『答案方向』畫一份報表，配文字『用量得到、講得清楚的特徵估價』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name C_fig1_scqa --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · 品牌平均價怎麼做才不洩漏
用在第 4 張。目的：全課最容易錯的一題，總整要再示範一次正確做法。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，一條由左至右的四步流程。第一步『整份二手車資料』。第二步『先切分』，畫出訓練與測試兩個色塊。第三步『只用訓練集依品牌算平均價』，畫出一張品牌對平均價的小表格，並用紅色虛線框住測試集標註『不參與計算』。第四步『把這張表併回訓練集與測試集』，兩個色塊都接上新欄位，標註『測試集只是查表』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name C_fig2_group_mean_safe --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
