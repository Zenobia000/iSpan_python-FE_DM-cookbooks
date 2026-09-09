# 🎨 概念示意圖提示詞 — 模組8 時間序列特徵工程

> 對應講義：[`M8_Time_Series_Feature_Engineering.md`](../../modules/module_08_time_series/slides/M8_Time_Series_Feature_Engineering.md)
> 生成模型：gpt-image-2（draw skill）｜風格：扁平向量教學插畫、含繁體中文標籤
> 規劃 3 張：時序切分方式 / lag 與各種窗 / 窗看到未來的兩種方式
>
> ⚠️ 含中文字的圖，gpt-image-2 偶有錯字或糊字 → 已預設 `--quality low`；標籤要完全清晰再改 `--quality high`。
> ▶️ 執行前先 `cd` 到本資料夾，圖會輸出到 `./concept_images/`：

```bash
cd data_mining_course/slides_plan/M08_時間序列
```

---

### 圖 1 · 時序的切分方式不一樣
用在第 1 張。目的：讓學員看見隨機切分在時序上為什麼是錯的。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，上下兩排時間軸。上排標紅色叉，一條時間軸上訓練與測試的色塊交錯混雜，標註『隨機切分：用未來預測過去』。下排標綠色勾，同一條時間軸左段整段是訓練色、右段整段是測試色，中間一條垂直分隔線標註『切分點』，並在右側標『測試集永遠比較晚』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M8_fig1_time_split --size 1536x1024 --quality low --outdir concept_images
```

### 圖 2 · lag 與各種窗
用在第 3 張。目的：一張圖把 lag、滾動窗、擴張窗的差別講完。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，三排時間序列的格子。第一排『lag 落後值』，一個箭頭從第 t-3 格指向第 t 格，標註『把三期前的值搬過來』。第二排『rolling 滾動窗』，一個固定寬度的框在格子上向右滑動，標註『固定長度，只涵蓋過去』。第三排『expanding 擴張窗』，框從左端起算逐漸變長，標註『從頭累積到現在』。三排右側都標一個紅色禁止符號指向 t 之後的格子，寫『不准碰』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M8_fig2_windows --size 1536x1024 --quality low --outdir concept_images
```

### 圖 3 · 窗看到未來的兩種方式
用在第 7 張。目的：把兩種最常見的未來洩漏並排。

```bash
python3 ~/.claude/skills/draw/draw.py "扁平向量教學插畫，白色背景，左右兩個錯誤案例都標紅色叉。左案『先洗牌再 shift』，一排時間格子被打散重排，箭頭從一格指到另一格但時間順序已亂，標註『shift 出來的不是前一期』。右案『置中窗』，一個框以當前格為中心同時涵蓋左右兩側，右半邊用紅色標示，標註『用到還沒發生的資料』。下方一行共同結論『先排序、按時間切、窗只往左』。柔和粉彩、扁平 infographic、繁體中文標籤清晰。" --name M8_fig3_future_leak --size 1536x1024 --quality low --outdir concept_images
```

---
> 風格微調：想更活潑可加「等角 isometric」；想要手繪感可改「whiteboard 手繪 doodle 風格」。
