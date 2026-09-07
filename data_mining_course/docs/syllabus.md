# 課程大綱

開課以 [course_charter.md](course_charter.md) 與 [lecture_map.md](lecture_map.md) 為準：**先模組工具，後心法總整。**

## 目標

- 會用特徵工程工具箱，並知道每個轉換在服務什麼假設。
- 表格走 domain driven；文／圖／聲／影走「理解 → 清理 → 導入 → 訓練」。
- 已會的 ML／Pandas 標前情複習。AI 工具可寫轉換；人盯 Why、切分、洩漏。
- 示範用真實資料。最後用英國二手車把心法走一遍。

## 上課順序

| 段 | 內容 | 預設 |
|---|---|---|
| 前情複習 | Module 1 EDA、Module 2 清理 | **自學** |
| 表格工具 | Module 3–8 | **上課** |
| 探勘落地 | Module 10（樹、端到端、分群；關聯規則一小段） | **上課** |
| AI 前處理 | Module 9、Module 11 | **上課** |
| 總整 | `capstone/capstone_used_car.ipynb` | **上課最後** |

## 完整技法庫

1. 導入與 EDA（自學複習）
2. 資料清理（自學複習）
3. 缺值與異常（House Prices）
4. 類別編碼（Titanic 只當編碼繞回）
5. 縮放與轉換（Insurance）
6. 特徵創造（NYC Taxi；日曆主角在此）
7. 特徵選擇與降維
8. 時間序列（lag／rolling）
9. 多模態前處理（真實文／圖／聲／影）
10. 探勘落地：Telco 樹重要性與端到端、Mall 分群；Apriori 對照現代推薦、Instacart 選做
11. 訓練前處理之後：資料格式 → 模型；LoRA／影片可教
