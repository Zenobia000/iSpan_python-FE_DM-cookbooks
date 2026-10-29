import numpy as np
from sklearn.metrics import mean_squared_error, accuracy_score

def evaluate_regression(y_true, y_pred):
    """評估迴歸模型"""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return {
        'MSE': mse,
        'RMSE': rmse
    }
