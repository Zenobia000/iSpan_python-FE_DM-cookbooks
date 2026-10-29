# 第 8 章　時間序列特徵工程：投影片規劃

9 張，採 `handout_blueprint.md` 的四格骨架（道 1 · 術 3 · 法 3 · 器 2）。
講義本體 [`M8_Time_Series_Feature_Engineering.md`](../../modules/module_08_time_series/slides/M8_Time_Series_Feature_Engineering.md)，產圖提示詞在 [`slides.prompts.md`](slides.prompts.md)，圖檔輸出到 `concept_images/`。

落在六步的第 5 到 6 步。這堂對應的鐵律是鐵律一的時序版（按時間切，不准洗牌）。

| # | 格 | 標題 | 這張要呈現什麼 | 對應 notebook | 圖 |
|---|---|---|---|---|---|
| 1 | 道 | 問題定位 | 時間序列把鐵律一換了一種形式：切分單位是時間，測試集必須是比較晚的一段 | `01_lag_features.ipynb` | 圖 1 |
| 2 | 術 | 第一原理 | 窗只能看過去。任何用到當下或未來資料的窗，都會讓模型在真實情境失效 | `01_lag_features.ipynb` | |
| 3 | 術 | 決策表 | lag／rolling／expanding／季節性分解四類，比看多久以前、參數、會不會不小心看到未來 | `02_rolling_windows.ipynb` | 圖 2 |
| 4 | 術 | 判斷表 | 依預測目標與資料頻率決定窗長；為什麼 lag 要先確認預測時點拿得到那個值 | `03_date_time_features.ipynb` | |
| 5 | 法 | SOP 五步 | 先排序 → 按時間切 → 造 lag 與窗 → 檢查窗的邊界 → 用時序交叉驗證 | `05_power_consumption_case.ipynb` | |
| 6 | 法 | 驗收三問 | 人話、洩漏、拿掉會怎樣。第二問要改寫成：這個窗有沒有看到預測時點還拿不到的資料 | — | |
| 7 | 法 | 反例 | 洗牌之後才做 `shift`，或者窗設成置中，兩種都會看到未來。附錯誤程式 | `02_rolling_windows.ipynb` | 圖 3 |
| 8 | 器 | 工具速查 | `shift`、`rolling`、`expanding`、`TimeSeriesSplit`、`seasonal_decompose` 的最小範例 | `04_seasonality_trend.ipynb` | |
| 9 | 器 | 三個陷阱與 notebook 對照 | 陷阱：用 `train_test_split` 洗牌切時序、窗置中、忘了先排序 | 全部 | |

要學員記住的那句話：窗只能看過去，這是時間序列唯一不能妥協的事。

對應 notebook 都在 `../../modules/module_08_time_series/notebooks/` 底下。
