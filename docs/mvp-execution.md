# Scientific Analysis MVP：執行與驗證紀錄

日期：2026-10-29。分支：`feature/ai-feature-decision-workshop`。

## 已交付與未交付

MVP 已實作單一 skill、本機 CLI、二手車九品牌唯讀載入、群組隔離 CV、三個核准衍生特徵、
固定參數 Dummy／Ridge／淺樹／LightGBM 比較、OOF 誤差、每 Fold 係數與樹規則、
離線 Plotly 互動報告，以及選取／篩選／註記的研究續接。

一般 CSV 的分類程式已有基本入口，但尚未完成代表案例驗收；不得視為完整分類交付。
自動調參、SHAP、分群／異常模型、一次 Holdout finalize、醫療與真實半導體研究驗證仍未實作。
不宣稱已滿足 80% 題目、找到根因、或完成獨立外部驗證。

## 設計裁決

- Ruling：MVP 採三 Fold 的開發集群組交叉驗證，不另保留 Holdout。理由是先驗證研究閉環；代價是模型選擇與探索結果不屬獨立確認，報告明示 development CV only。
- Ruling：保留九品牌主檔，以正規化後非目標屬性的相同列建立 proxy group；不是已驗證的車輛 ID，也不保證完全消除來源重複。
- Ruling：2020 是既有課程快照假設，未來年份拒絕計算車齡；不以當前年份默算車齡。
- Ruling：特徵需 `approved_features=true`，拒絕靜默核准；原始特徵模型可不執行 features 階段。
- Ruling：專家後驗選出的子群保留父 run 與完整資料，另外計算子群摘要；不把子群當新獨立樣本或自動重訓。
- Ruling：HTML 只顯示最多 20 個絕對值最大的係數，完整係數保留 JSON，明示轉換尺度與共線性限制。

## 環境與命令

```bash
uv sync --extra harness
uv run --extra harness pytest tests/harness -q
uv run --extra harness pytest .agents/skills/scientific-analysis/tests -q
```

建立環境前沒有 `.venv`；uv 在專案內建立虛擬環境，沒有使用系統 pip。
測試 Python 為 3.11。`pyproject.toml` 與 `uv.lock` 同步加入 harness extra。
Playwright 使用本機 Chromium，僅執行本地報告，未上傳資料。

真實資料試跑的規格存於被忽略的 `workspace/mvp-real-spec-003.json`，內容為九品牌二手車、
12,000 列群組保留抽樣、seed=42、folds=3、snapshot_year=2020、approved_features=true。

```bash
for stage in inspect explore features model explain report; do
  uv run --extra harness python .agents/skills/scientific-analysis/scripts/run.py "$stage" \
    --spec workspace/mvp-real-spec-003.json || break
done
```

新環境應使用 skill 的 `examples/cars.json` 建立自己的新 run；工作目錄中的試跑檔不進版控。
各階段拒絕覆寫既有產物；首次失敗或程式格式變更時建立新 run，沒有重寫舊結果。

## 測試證據

- Red：實作前執行 core 測試，因不存在 core.py 而失敗。
- Green：核心測試覆蓋目標不可進衍生公式、未來年份拒絕、群組隔離、完整群組抽樣、Fold 內補值與未知類別、非有限 JSON、拒絕覆寫、錯資料版本／列 ID、篩選自行驗算。
- CLI 整合：人工 fixture 走 inspect／explore／features／model／explain／report／resume，確認原檔不變、配對 OOF 列數、篩選而未框選時仍得到正確子群摘要、split 被改動時拒絕後續執行。
- 真實九品牌試跑曾發現 Pandas boolean index 不對齊，已改用位置 mask 並補回歸測試。
- 瀏覽器測試涵蓋離線載入、圖表連動、空集合、匯出與續接狀態、單位與模型解釋呈現。
- 測試中的合成資料只用於軟體行為驗證，不冒充 Kaggle 研究結果。

既有依賴出現 pytest anyio deprecation 警告；LightGBM 對轉換後陣列的 feature-name 警告亦可見。
未以關閉警告代替檢查，這些警告不代表有額外獨立驗證。

## 真實資料的結論邊界

最後驗證產物：`workspace/used_cars/mvp-real-003/report.html`，12,000 列；
Ridge 首 Fold 的全部 182 個係數保存在 JSON，報告只顯示絕對值最高的 20 項。
續接產物：`workspace/used_cars/mvp-followup-003/report.html`，保留完整資料背景，
並恢復 BMW／里程 10,000–60,000 英里的篩選，共 661 列；續接的 profile 保存該子群摘要。
這個續接示範的條件由腳本建立，瀏覽器實際匯出／恢復另由 Playwright 測試驗證。

本機命令結果：核心測試 8 passed，互動報告測試 4 passed（由報告實作者執行）；
整合全套 12 passed 由主代理再次確認。沒有把測試通過等同領域專家驗收完成。

原始／新特徵使用相同列與 Fold；模型的得分差異只支持此開發集的預測比較。
LightGBM 的新特徵 MAE 改善幅度很小，不能因此宣稱特徵工程顯著成功或支持任何因果機制。
Ridge 與樹模型並列保留，沒有自動替換為分數最高者。

## 後續里程碑

1. 先由專家操作 MVP 的真實報告並完成一次研究往返，記錄理解／互動卡點。
2. 再做分類與有限調參、獨立驗收設計及 SHAP。
3. 之後才加入分群、異常與領域壓力測試。

本次沒有刪除舊程式、移動原始 CSV、commit、push 或 merge。
