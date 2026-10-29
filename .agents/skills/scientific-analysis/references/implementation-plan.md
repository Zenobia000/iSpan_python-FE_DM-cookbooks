# MVP 與後續驗證界線

本文件是目前實作的驗收方向，不宣稱所有項目已通過。真實執行證據見專案 `docs/mvp-execution.md`。

## MVP

以隨庫 UK Used Cars 九品牌主檔完成一輪研究：問題與假說、來源檢查、群組切分、互動探索、核准特徵、固定參數模型、解釋與報告。
首要目標是專家看圖後指出子群，agent 可驗證匯出狀態並建立第二個 run，保留前一輪證據。
完整操作參考 [研究往返](research-workflow.md)；演算法與圖形選擇參考 [方法卡](methods.md)。

## 介面與階段

`run.py <stage> --spec <file>`；階段為 inspect、explore、features、model、explain、report、resume。
inspect 建立新 run 及來源／Fold，後續階段依序寫自己的產物；存在就拒絕覆寫。
resume 接受 `--view <exported-view.json> --new-run-id <id>`，驗證來源後建立分支，不沿用舊模型分數冒充新分析。

spec 包含 case_id、run_id、data_path 或二手車 data_dir、task、target、group_column／time_column、snapshot_year、sample_n、seed、folds、approved_features、question、hypotheses、output_root。
task 選 regression、classification、clustering 或 anomaly；欄位必要性與實際限制以 CLI 驗證結果為準。
目前 MVP 的 clustering／anomaly 建模會明確拒絕，屬後續擴充；列出 task 名稱不是完整支援的承諾。
二手車範例預設不核准衍生特徵，需專家確認後修改規格並建立新 run。不能讓 agent 為通過流程自行勾選。

## 後續擴充

- 分類／回歸：可解釋基準與性能候選、限額調參、消融、校準與適用的 SHAP。
- 無監督：K-Means／DBSCAN、穩健異常／Isolation Forest、群組剖面及穩定性。
- 領域壓力測試：醫療正類／重複量測；工業批次／時間捷徑。人工反例不等於真實研究效益。

CLI 可接受某個 task，不代表其所有分析方法、圖表、統計驗證或領域效益已完成；每次交付須列真實執行範圍。

## 驗收

原檔 hash 不變、Fold 群組隔離、學得轉換只 fit 訓練 Fold、新舊特徵配對資料列一致、報告對回計算產物。
瀏覽器檢查離線使用、連動選取、篩選／高亮語意、空集合、重設、錯版匯回拒絕及註記文字安全。
操作閉環未成立時先修正互動與研究脈絡，不新增模型掩蓋問題。
