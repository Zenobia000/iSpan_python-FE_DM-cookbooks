# 封存索引

這裡保存原特徵工程課程與歷史素材。現行專案入口是 [根目錄 README](../README.md)，決策分析方法參考在 [docs/references/](../docs/references/README.md)。

本次整理保留檔案內容與舊課程的內部目錄層級，將與現行 Harness 無直接關係的內容集中收納。封存內容不是日常技能、開發或測試的預設入口。

## 收納內容

| 位置 | 內容 |
| --- | --- |
| [README-course.md](README-course.md) | 原課程首頁 |
| [data_mining_course/](data_mining_course/README.md) | 11 個模組、總整、專案、投影片規劃、課程環境與資料工具 |
| `course_slides/` | MLOps 平台比較原稿；特徵工程指南已移至現行方法參考 |
| `assets/` | 舊課程首頁圖片、動畫與生成腳本 |
| [UV_GUIDE.md](UV_GUIDE.md) | 舊課程的 uv 操作說明 |
| [GIT_IGNORE_GUIDE.md](GIT_IGNORE_GUIDE.md) | 舊課程的版控忽略說明 |
| [legacy-config/cursorrules.txt](legacy-config/cursorrules.txt) | 原 `.cursorrules`，以文字檔保存，不再放在根目錄作為規約入口 |
| `local/` | 原根目錄的未追蹤圖片與 Python 快取；僅在本機保留，不提交 |

## 路徑對照

| 原位置 | 新位置 |
| --- | --- |
| `data_mining_course/` | `archive/data_mining_course/` |
| `data_mining_course/docs/AI協作決策分析架構.md` | `docs/AI協作決策分析架構.md` |
| `course_slides/特徵工程系統化分析指南.pdf` | `docs/references/特徵工程系統化分析指南.pdf` |
| `course_slides/特徵工程系統化分析指南.docx` | `docs/references/特徵工程系統化分析指南.docx` |
| `course_slides/MLOps_平台比較分析與建議.docx` | `archive/course_slides/MLOps_平台比較分析與建議.docx` |
| `README.md`（原課程首頁） | `archive/README-course.md` |
| `assets/` | `archive/assets/` |
| `.cursorrules` | `archive/legacy-config/cursorrules.txt` |

## 使用封存教材

課程內部的相對層級保持原狀，本機已下載資料與快取也隨課程目錄移動，仍由 `.gitignore` 排除。既有隨教材版控的範例資料保留原追蹤範圍。

歷史文件中的根目錄執行指令與絕對路徑可能需要依新位置調整；本次未重新執行整套教材、下載資料或啟動模型訓練。若要維護教材，先讀 [封存協作指引](AGENTS.md)。
