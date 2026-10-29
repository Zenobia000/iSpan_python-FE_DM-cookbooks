# Module 8: Time Series Feature Engineering

## 這堂怎麼上

| | |
|---|---|
| 角色 | **上課**（主線：lag、rolling、時間切分） |
| 講義 | [`slides/M8_Time_Series_Feature_Engineering.md`](slides/M8_Time_Series_Feature_Engineering.md) |
| 資料 | 電力消耗繞回（`05`） |
| 不要跟誰重複 | **日曆拆欄已在 M6。** `03_date_time_features` 標自學複習／只留接窗特徵所需的最小時間欄。 |

Welcome to Module 8. In this module, we will dive into the specialized world of feature engineering for time series data. Time series data has unique characteristics, such as temporal dependence, trends, and seasonality, which require specific techniques to handle effectively.

## Learning Objectives

By the end of this module, you will be able to:

-   **Create Lag Features**: Understand and create lag features to capture the auto-correlation structure of time series data.
-   **Implement Rolling Window Features**: Generate features based on rolling windows (e.g., moving averages, standard deviations) to capture local trends and volatility.
-   **Extract Date and Time Features**: Decompose timestamps into meaningful components like year, month, day of the week, and hour.
-   **Analyze Seasonality and Trend**: Decompose time series into trend, seasonal, and residual components to better understand and model the data.
-   **Apply Techniques to a Real-World Dataset**: Integrate這些 all these techniques in a real-world case study using a power consumption dataset.

## Notebooks

1.  `01_lag_features.ipynb`: Introduction to creating lag features.
2.  `02_rolling_windows.ipynb`: Techniques for creating rolling window statistics.
3.  `03_date_time_features.ipynb`: Extracting valuable features from date and time stamps.
4.  `04_seasonality_trend.ipynb`: Decomposing time series to identify underlying patterns.
5.  `05_power_consumption_case.ipynb`: A comprehensive case study applying all the learned techniques. 