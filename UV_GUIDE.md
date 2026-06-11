# 🚀 uv 新手上手指南

> 給這堂課的學員：**第一次用 uv 也能五分鐘把環境跑起來。**
> 本文是 [README 的「快速開始」](README.md#-快速開始)的延伸版，補上安裝細節、心智模型、指令速查、疑難排解，以及給 pip / conda 老用戶的對照表。

---

## 📋 目錄

- [1. uv 是什麼？為什麼這堂課用它](#1-uv-是什麼為什麼這堂課用它)
- [2. 最重要的觀念：忘掉 `activate`](#2-最重要的觀念忘掉-activate)
- [3. 安裝 uv（一次就好）](#3-安裝-uv一次就好)
- [4. 三步驟啟動本課程](#4-三步驟啟動本課程)
- [5. 這個專案的依賴分層](#5-這個專案的依賴分層)
- [6. 日常指令速查](#6-日常指令速查)
- [7. 疑難排解 FAQ](#7-疑難排解-faq)
- [8. 給 pip / conda 老用戶的對照表](#8-給-pip--conda-老用戶的對照表)
- [9. 參考連結](#9-參考連結)

---

## 1. uv 是什麼？為什麼這堂課用它

**一句話**：uv 是用 Rust 寫的、目前最快的 Python 工具，一個指令就取代了 `pip` + `venv` + `virtualenv` + `pyenv` + `pip-tools` + `pipx` 一整套。

這堂課用它的理由很單純 —— **可重現**：

- `pyproject.toml`：**唯一**的依賴來源（你要裝什麼套件，只看這份）。
- `uv.lock`：把每個套件鎖到**精確版本**。
- 結果：你、老師、同學、Docker、CI 裝出來的環境**位元級一致**，不會再有「我電腦跑得動、你電腦壞掉」的鬼故事。

---

## 2. 最重要的觀念：忘掉 `activate`

用 uv 之後，你**不用**自己建虛擬環境、**不用** `source .venv/bin/activate`。只要記住兩個動詞：

| 你想做的事 | 指令 |
| :--- | :--- |
| 🏗️ **裝好/還原環境** | `uv sync` |
| ▶️ **在環境裡跑東西** | `uv run <任何指令>` |

例如：`uv run jupyter lab`、`uv run python data_download.py`。
`uv run` 每次都會先確認環境跟 `uv.lock` 一致，再執行 —— 所以你永遠跑在對的環境裡，**不會忘記 activate**。

> 💡 真的想進傳統的 activate 模式也行：`source .venv/bin/activate`（Windows：`.venv\Scripts\activate`）。但課程一律建議用 `uv run`，少一個出錯點。

---

## 3. 安裝 uv（一次就好）

**macOS / Linux：**
```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

**Windows（PowerShell）：**
```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

**驗證安裝：**
```bash
uv --version        # 例如 uv 0.11.20
```

> ⚠️ 裝完若提示 `uv: command not found`，多半是 PATH 還沒更新 —— **關掉終端機重開**一次即可（uv 預設裝在 `~/.local/bin`）。

之後想升級 uv 自己：`uv self update`。
（替代安裝法：`pip install uv` 或 `pipx install uv`。）

---

## 4. 三步驟啟動本課程

```bash
# 1️⃣ 取得專案
git clone https://github.com/Zenobia000/iSpan_python-FE_DM-cookbooks.git
cd iSpan_python-FE_DM-cookbooks

# 2️⃣ 建立環境（自動建立 .venv、照 uv.lock 還原核心依賴）
uv sync

# 3️⃣ 開始上課
uv run jupyter lab
```

`uv sync` 這一步會自動幫你：

1. 找一個符合 `requires-python`（**Python 3.9 – 3.12**）的直譯器，沒有就**自動下載**一個。
2. 在專案資料夾建立 `.venv/`。
3. 照 `uv.lock` 把套件**精確還原**。

> 📦 下載課程資料集需要先設定 Kaggle API，步驟見 [README → 配置 Kaggle API](README.md#3️⃣-配置-kaggle-api)，之後 `uv run python data_download.py`。

---

## 5. 這個專案的依賴分層

`pyproject.toml` 把套件分成三層，照你的需求裝：

| 層級 | 內容 | 何時需要 | 指令 |
| :--- | :--- | :--- | :--- |
| **核心** | pandas、numpy、scikit-learn、matplotlib、xgboost、lightgbm、jupyterlab… | 模組 **M01–M08、M10** | `uv sync` |
| **multimodal**（選用） | tensorflow、librosa、nltk、opencv-python、scikit-image、spacy | 模組 **M09**（文字／圖像／音訊，套件較重） | `uv sync --extra multimodal` |
| **dev**（預設裝） | jupytext、pdf2docx | 維護課程、轉檔時 | 預設隨 `uv sync` 一起裝 |

```bash
uv sync                      # 只裝核心（最輕，先跑 M01–M08/M10 夠用）
uv sync --extra multimodal   # 要做 M09 多模態時再加
uv sync --all-extras         # 一次全部裝齊
uv sync --no-dev             # 不要 dev 維護工具
```

> 💡 `--extra` 是**選用**的、預設不裝，要手動加；`dev` 群組則是**預設會裝**，不要才加 `--no-dev`。

---

## 6. 日常指令速查

| 想做的事 | 指令 |
| :--- | :--- |
| 還原 / 更新環境（照 lock） | `uv sync` |
| 在環境裡執行任何東西 | `uv run <指令>`（如 `uv run jupyter lab`） |
| 新增一個套件 | `uv add seaborn` |
| 加到某選用群組 / dev 群組 | `uv add --optional multimodal opencv-python`<br>`uv add --group dev ruff` |
| 移除套件 | `uv remove seaborn` |
| 重算鎖檔 | `uv lock`（全部升級：`uv lock --upgrade`） |
| 看依賴樹 | `uv tree` |
| 安裝 / 指定 Python 版本 | `uv python install 3.12`<br>`uv python pin 3.12`（寫入 `.python-version`） |
| 一次性跑 CLI 工具（像 pipx） | `uvx ruff check .` |
| 常駐安裝 CLI 工具 | `uv tool install ruff` |
| 升級 uv 本身 | `uv self update` |

---

## 7. 疑難排解 FAQ

<details>
<summary><b>Q：輸入 <code>uv</code> 說 command not found？</b></summary>

PATH 還沒生效。**關掉終端機重開**，或 `source ~/.bashrc`（zsh 用 `~/.zshrc`）。uv 預設在 `~/.local/bin`。
</details>

<details>
<summary><b>Q：想用特定的 Python 版本？</b></summary>

```bash
uv python install 3.12     # 下載 3.12
uv python pin 3.12         # 在本專案鎖定 3.12（產生 .python-version）
uv sync                    # 用新版本重建環境
```
本課程支援 3.9–3.12，遇到套件相容性問題時，**3.11 / 3.12 最穩**。
</details>

<details>
<summary><b>Q：M09 的 tensorflow / opencv 裝失敗？</b></summary>

1. 先確認 Python 是 **3.9–3.12**（這些套件對版本敏感）。
2. 先別裝 multimodal，用 `uv sync` 把核心跑起來，確認 M01–M08 沒問題。
3. 再單獨 `uv sync --extra multimodal`，看是哪個套件報錯。
</details>

<details>
<summary><b>Q：想砍掉環境重來？</b></summary>

```bash
rm -rf .venv && uv sync     # 刪掉虛擬環境，照 lock 乾淨重建
```
不會動到你的程式碼，只重建 `.venv`。
</details>

<details>
<summary><b>Q：我已經有 conda / Anaconda，會衝突嗎？</b></summary>

不會。uv 在專案資料夾建**獨立的 `.venv`**，跟你的 conda base 完全隔離。建議上課時別在 conda 環境裡，直接用 `uv run` 即可。
</details>

<details>
<summary><b>Q：VS Code / Jupyter 找不到正確的 kernel？</b></summary>

- VS Code：右下角選直譯器 → 指到 `.venv/bin/python`（Windows：`.venv\Scripts\python.exe`）。
- 或一律用 `uv run jupyter lab` 啟動，它本來就跑在對的環境。
</details>

---

## 8. 給 pip / conda 老用戶的對照表

| 舊習慣 | uv 對應 |
| :--- | :--- |
| `python -m venv .venv` | （免）`uv sync` 自動建 |
| `source .venv/bin/activate` | （免）改用 `uv run` |
| `pip install -r requirements.txt` | `uv sync` |
| `pip install pandas` | `uv add pandas` |
| `pip uninstall pandas` | `uv remove pandas` |
| `pip freeze > requirements.txt` | （免）`uv.lock` 自動維護；要匯出再 `uv export` |
| `conda create -n env python=3.12` | `uv python pin 3.12` + `uv sync` |
| `python script.py` | `uv run python script.py` |
| `pipx run black` | `uvx black` |

> 📌 **本專案請一律用 `uv add` / `uv sync` 管理套件**，不要在專案裡混用 `pip install`，以免 `uv.lock` 與實際環境不一致。

---

## 9. 參考連結

- 官方文件（完整說明書）：<https://docs.astral.sh/uv/>
- 快速上手：<https://docs.astral.sh/uv/getting-started/>
- 指令參考：<https://docs.astral.sh/uv/reference/cli/>
- 本機離線說明：`uv help` 或 `uv <子指令> --help`（例：`uv sync --help`）

---

> 🙋 卡關了？先看上面的 [FAQ](#7-疑難排解-faq)，再不行就到專案 [Issues](../../issues) 發問，附上 `uv --version`、`python --version` 與完整錯誤訊息。
