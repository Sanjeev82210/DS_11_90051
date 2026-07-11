import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate_model(y_true, y_pred):
    """Calculate MAE, RMSE, R² for a prediction."""
    return {
        "MAE": mean_absolute_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "R2": r2_score(y_true, y_pred),
    }


def evaluate_model_detailed(y_true, y_pred):
    """Extended metrics including MAPE where possible."""
    base = evaluate_model(y_true, y_pred)
    y_true_arr = np.asarray(y_true, dtype=float)
    y_pred_arr = np.asarray(y_pred, dtype=float)
    mask = y_true_arr != 0
    if mask.sum() > 0:
        base["MAPE"] = np.mean(np.abs((y_true_arr[mask] - y_pred_arr[mask]) / y_true_arr[mask])) * 100
    else:
        base["MAPE"] = np.nan
    base["Max Error"] = np.max(np.abs(y_true_arr - y_pred_arr))
    return base
