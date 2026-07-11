import pandas as pd
import numpy as np
import warnings
import os
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from statsmodels.tsa.arima.model import ARIMA
from sklearn.model_selection import TimeSeriesSplit
import joblib
import sys

# Ensure src is on the path
sys.path.insert(0, os.path.dirname(__file__))
from evaluation import evaluate_model


def get_models():
    """Return the four baseline model instances."""
    return {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, random_state=42
        ),
        "XGBoost": XGBRegressor(
            n_estimators=200, learning_rate=0.05,
            max_depth=4, random_state=42
        ),
    }


def train_and_evaluate(X, y, tscv_splits=5):
    """Run TimeSeriesSplit cross-validation on all models including ARIMA.
    Returns a DataFrame of average metrics per model *and* a dict of
    {model_name: (fitted_model, y_test, predictions)} from the *last* fold
    for plotting purposes.
    """
    tscv = TimeSeriesSplit(n_splits=tscv_splits)
    models = get_models()

    results = []
    last_fold_data = {}  # for Actual vs Predicted chart

    for name, model in models.items():
        fold_metrics = []
        for train_idx, test_idx in tscv.split(X):
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            fold_metrics.append(evaluate_model(y_test, preds))

        # Save last fold for plotting
        last_fold_data[name] = {
            "y_test": y.iloc[test_idx].values,
            "y_pred": preds,
            "years": X.iloc[test_idx]["year"].values if "year" in X.columns else None,
        }

        avg = {
            "Model": name,
            "MAE": np.mean([m["MAE"] for m in fold_metrics]),
            "RMSE": np.mean([m["RMSE"] for m in fold_metrics]),
            "R2": np.mean([m["R2"] for m in fold_metrics]),
        }
        results.append(avg)

    # ARIMA / ARIMAX
    fold_metrics = []
    for train_idx, test_idx in tscv.split(X):
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                arima = ARIMA(endog=y_train, exog=X_train, order=(1, 1, 1))
                fitted = arima.fit()
                preds = fitted.forecast(steps=len(y_test), exog=X_test)
                fold_metrics.append(evaluate_model(y_test, preds))
        except Exception:
            pass

    if fold_metrics:
        last_fold_data["ARIMA"] = {
            "y_test": y.iloc[test_idx].values,
            "y_pred": preds,
            "years": X.iloc[test_idx]["year"].values if "year" in X.columns else None,
        }
        results.append({
            "Model": "ARIMA",
            "MAE": np.mean([m["MAE"] for m in fold_metrics]),
            "RMSE": np.mean([m["RMSE"] for m in fold_metrics]),
            "R2": np.mean([m["R2"] for m in fold_metrics]),
        })

    return pd.DataFrame(results), last_fold_data


def train_final_model(X, y, model_name="XGBoost"):
    """Train on all data and persist."""
    models = get_models()
    model = models.get(model_name)
    if model is None:
        raise ValueError(f"Model '{model_name}' not found.")
    model.fit(X, y)
    os.makedirs("models", exist_ok=True)
    path = f"models/{model_name.replace(' ', '_').lower()}_model.pkl"
    joblib.dump(model, path)
    return model


def get_feature_importance(model, feature_names):
    """Extract feature importance for tree-based or linear models."""
    if hasattr(model, "feature_importances_"):
        imp = model.feature_importances_
    elif hasattr(model, "coef_"):
        imp = np.abs(model.coef_)
    else:
        return None
    return pd.DataFrame({
        "Feature": feature_names,
        "Importance": imp,
    }).sort_values("Importance", ascending=False)


if __name__ == "__main__":
    from features import create_features, get_feature_list

    climate = pd.read_csv("data/processed/climate_data.csv")
    df = create_features(climate)
    features = get_feature_list()
    X, y = df[features], df["temperature_anomaly"]

    results_df, last_fold = train_and_evaluate(X, y)
    print("Cross-Validation Results:")
    print(results_df.to_string(index=False))

    print("\nTraining final XGBoost model...")
    model = train_final_model(X, y, "XGBoost")
    fi = get_feature_importance(model, features)
    print("\nFeature Importance:")
    print(fi.to_string(index=False))
