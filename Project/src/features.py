import pandas as pd
import numpy as np


def create_features(df):
    """Engineer time-series features from the merged climate dataset."""
    df = df.copy()
    df = df.sort_values("year").reset_index(drop=True)

    # --- Temporal features ---
    df["years_since_start"] = df["year"] - df["year"].min()
    df["decade"] = (df["year"] // 10) * 10

    # --- CO2 features ---
    df["co2_growth"] = df["co2_emissions"].pct_change() * 100
    df["cumulative_co2"] = df["co2_emissions"].cumsum()
    df["co2_ppm_change"] = df["co2_ppm"].diff()

    # --- Temperature rolling averages ---
    df["temp_5yr_avg"] = df["temperature_anomaly"].rolling(5, min_periods=1).mean()
    df["temp_10yr_avg"] = df["temperature_anomaly"].rolling(10, min_periods=1).mean()

    # --- Lag features ---
    df["co2_lag_1"] = df["co2_ppm"].shift(1)
    df["co2_lag_5"] = df["co2_ppm"].shift(5)
    df["temp_lag_1"] = df["temperature_anomaly"].shift(1)

    # --- Rate of change ---
    df["temp_rate_of_change"] = df["temperature_anomaly"].diff()

    # Drop NaN rows created by shift/lag (keep rolling averages)
    df = df.dropna(subset=["co2_lag_1", "co2_lag_5", "temp_lag_1"]).reset_index(drop=True)
    return df


def get_feature_list():
    """Return the canonical list of model input features."""
    return [
        "year", "co2_ppm", "cumulative_co2",
        "co2_emissions", "co2_lag_1", "co2_lag_5"
    ]


if __name__ == "__main__":
    df = pd.read_csv("data/processed/climate_data.csv")
    df_features = create_features(df)
    print("Features created.")
    print(df_features.columns.tolist())
    print(f"Total rows: {len(df_features)}")
    print(df_features.head())
