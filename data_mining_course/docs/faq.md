# 常見問題解答

## 環境設定
**Q: 如何安裝所需套件？**
A: 請參考 `environment` 資料夾中的 `requirements.txt` 或 `environment.yml` 檔案，使用 pip 或 conda 進行安裝。例如：`pip install -r requirements.txt`。

## 執行與路徑
**Q: 為什麼我的筆記本 (notebook) 找不到資料集檔案？出現 `FileNotFoundError`。**
A: 所有的筆記本都預設您是從 **專案的根目錄** (`iSpan_python-FE_DM-cookbooks/`) 來執行它們的。

- **若您使用 VS Code**: 請確保您的終端機 (terminal) 的當前工作目錄位於專案根目錄。
- **若您使用 Jupyter Lab/Notebook**: 請從專案根目錄啟動 Jupyter 服務 (`jupyter lab`)。

我們使用的相對路徑（如 `data_mining_course/datasets/...`）是基於這個前提。如果您從 `notebooks` 子目錄中執行，就會因為相對路徑錯誤而找不到檔案。