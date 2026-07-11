"""
src/explainability.py
Helper functions for model explainability (SHAP) and scenario threshold analysis.
"""
import numpy as np
import pandas as pd
import warnings


def compute_shap_values(model, X):
    """
    Compute SHAP values for a tree-based model (XGBoost/RandomForest).
    Returns (shap_values_array, explainer) or (None, None) on failure.
    """
    try:
        import shap
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            explainer = shap.TreeExplainer(model)
            shap_values = explainer.shap_values(X)
        return shap_values, explainer
    except Exception:
        return None, None


def shap_summary_df(shap_values, feature_names):
    """
    Returns a DataFrame with mean absolute SHAP value per feature, sorted descending.
    """
    if shap_values is None:
        return pd.DataFrame()
    mean_abs = np.abs(shap_values).mean(axis=0)
    return (
        pd.DataFrame({"Feature": feature_names, "Mean |SHAP|": mean_abs})
        .sort_values("Mean |SHAP|", ascending=False)
        .reset_index(drop=True)
    )


def get_threshold_crossing_year(temp_series, year_series, threshold):
    """
    Given parallel arrays of temperature anomaly and year, returns the first year
    where temperature_anomaly >= threshold, or None if never crossed.
    """
    for temp, yr in zip(temp_series, year_series):
        if temp >= threshold:
            return int(yr)
    return None


def build_policy_tracker(all_scenarios_df, thresholds=(1.5, 2.0)):
    """
    For each scenario in all_scenarios_df (columns: scenario, year, temperature_anomaly),
    find the crossing year for each threshold.

    Returns a DataFrame: scenario | 1.5°C Crossing | 2.0°C Crossing
    """
    rows = []
    for scenario_name, grp in all_scenarios_df.groupby("scenario"):
        grp = grp.sort_values("year")
        row = {"Scenario": scenario_name}
        for t in thresholds:
            yr = get_threshold_crossing_year(
                grp["temperature_anomaly"].values,
                grp["year"].values,
                t,
            )
            col = f"{t}\u00b0C Crossing"
            row[col] = str(yr) if yr else "Does not cross"
        rows.append(row)
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import sys
    sys.path.insert(0, ".")
    from features import create_features, get_feature_list
    from models import train_final_model
    from scenarios import compare_scenarios

    df = create_features(__import__("pandas").read_csv("data/processed/climate_data.csv"))
    features = get_feature_list()
    X, y = df[features], df["temperature_anomaly"]
    model = train_final_model(X, y, "XGBoost")

    shap_vals, _ = compute_shap_values(model, X)
    summary = shap_summary_df(shap_vals, features)
    print("SHAP Summary:\n", summary)

    all_sc = compare_scenarios(df, 2100, features, model)
    tracker = build_policy_tracker(all_sc)
    print("\nPolicy Tracker:\n", tracker)
