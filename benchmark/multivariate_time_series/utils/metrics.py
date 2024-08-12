import numpy as np

def metric(preds, trues):
    mae = np.mean(np.abs(preds - trues))
    mse = np.mean((preds - trues) ** 2)
    rmse = np.sqrt(mse)
    mape = np.mean(np.abs((preds - trues) / trues)) * 100
    return mae, mse, rmse, mape
