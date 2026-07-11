import pandas as pd
import numpy as np

SCENARIOS = {
    "🔴 High Emissions (+2%/yr)": {
        "annual_co2_change": 0.02,
        "color": "#EF4444",
        "description": "Emissions continue to grow at ~2% per year, following recent fossil-fuel expansion trends.",
    },
    "🟡 Current Policies (+0.5%/yr)": {
        "annual_co2_change": 0.005,
        "color": "#F59E0B",
        "description": "Emissions plateau with current policies in place — small but continued growth.",
    },
    "🟢 Moderate Reduction (−2%/yr)": {
        "annual_co2_change": -0.02,
        "color": "#10B981",
        "description": "Countries implement moderate decarbonization in line with existing Paris pledges.",
    },
    "🔵 Aggressive Reduction (−5%/yr)": {
        "annual_co2_change": -0.05,
        "color": "#3B82F6",
        "description": "Rapid, aggressive transition to renewables — close to 1.5 °C pathway.",
    },
}


def generate_scenario(df, start_year, end_year, annual_change):
    """Project future CO₂ pathway & generate model-ready features."""
    last_row = df.iloc[-1]

    co2_emissions = float(last_row["co2_emissions"])
    cumulative_co2 = float(last_row["cumulative_co2"])
    co2_ppm = float(last_row["co2_ppm"])

    # Keep recent ppm history for lag features
    recent_co2_ppm = df["co2_ppm"].tail(5).tolist()

    future = []
    for year in range(start_year, end_year + 1):
        co2_emissions *= (1 + annual_change)
        cumulative_co2 += co2_emissions

        # ~2.3 ppm per 10 Gt CO₂ (airborne fraction ≈ 44%)
        ppm_increase = co2_emissions / 7800.0
        co2_ppm += ppm_increase

        recent_co2_ppm.append(co2_ppm)

        future.append({
            "year": year,
            "co2_emissions": co2_emissions,
            "cumulative_co2": cumulative_co2,
            "co2_ppm": co2_ppm,
            "co2_lag_1": recent_co2_ppm[-2],
            "co2_lag_5": recent_co2_ppm[-6],
        })

    return pd.DataFrame(future)


def compare_scenarios(df, end_year, feature_cols, model):
    """Run all built-in scenarios and return combined results for charting."""
    last_year = int(df["year"].max())
    frames = []

    for name, cfg in SCENARIOS.items():
        future = generate_scenario(df, last_year + 1, end_year, cfg["annual_co2_change"])
        preds = model.predict(future[feature_cols])
        future["temperature_anomaly"] = preds
        future["scenario"] = name
        frames.append(future)

    return pd.concat(frames, ignore_index=True)


if __name__ == "__main__":
    from features import create_features

    climate = pd.read_csv("data/processed/climate_data.csv")
    df = create_features(climate)

    future_df = generate_scenario(
        df, 2018, 2050,
        SCENARIOS["🔵 Aggressive Reduction (−5%/yr)"]["annual_co2_change"],
    )
    print("Aggressive Reduction Scenario:")
    print(future_df.head())
    print(future_df.tail())
