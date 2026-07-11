import pandas as pd
import numpy as np


def load_temperature_data(path):
    """Load and process HadCRUT monthly temperature anomaly data to annual."""
    df = pd.read_csv(path, header=None)
    # Columns: 0=date, 1=median anomaly, 2-11=confidence bounds etc
    df = df[[0, 1]]
    df.columns = ["date", "temperature_anomaly"]
    df["temperature_anomaly"] = pd.to_numeric(df["temperature_anomaly"], errors="coerce")
    df["year"] = pd.to_datetime(df["date"], errors="coerce").dt.year
    df = df.dropna(subset=["year", "temperature_anomaly"])
    df["year"] = df["year"].astype(int)

    # Aggregate monthly to annual mean
    df_annual = df.groupby("year")["temperature_anomaly"].mean().reset_index()
    return df_annual


def load_co2_data(path):
    """Load OWID CO2 emissions data, extracting global totals."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    df = df.drop_duplicates()

    # Global aggregate
    if "country" in df.columns:
        df_global = df[df["country"] == "World"].copy()
    else:
        df_global = df.copy()

    keep_cols = ["year", "co2", "co2_per_capita", "population", "gdp"]
    df_global = df_global[[c for c in keep_cols if c in df_global.columns]]
    df_global = df_global.rename(columns={"co2": "co2_emissions"})
    df_global["year"] = pd.to_numeric(df_global["year"], errors="coerce")
    df_global = df_global.dropna(subset=["year"])
    df_global["year"] = df_global["year"].astype(int)
    return df_global.drop_duplicates(subset=["year"])


def load_co2_country_data(path):
    """Load OWID CO2 data with country-level detail for country analysis."""
    df = pd.read_csv(path, low_memory=False)
    df.columns = df.columns.str.lower().str.strip()
    df = df.drop_duplicates()

    # Filter out aggregates (World, continents, etc.)
    aggregates = [
        "World", "Asia", "Europe", "North America", "South America",
        "Africa", "Oceania", "High-income countries",
        "Low-income countries", "Lower-middle-income countries",
        "Upper-middle-income countries", "European Union (27)",
        "European Union (28)", "Asia (excl. China and India)",
        "Europe (excl. EU-27)", "Europe (excl. EU-28)",
        "North America (excl. USA)", "International transport",
    ]
    df = df[~df["country"].isin(aggregates)]
    df = df.dropna(subset=["iso_code"])  # Drop entries without ISO code

    keep_cols = ["country", "iso_code", "year", "co2", "co2_per_capita",
                 "population", "gdp", "cumulative_co2"]
    df = df[[c for c in keep_cols if c in df.columns]]
    df = df.rename(columns={"co2": "co2_emissions"})
    df["year"] = pd.to_numeric(df["year"], errors="coerce")
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)
    return df


def load_atmospheric_co2_data(path):
    """Load Mauna Loa atmospheric CO2 and aggregate to annual."""
    df = pd.read_csv(path)
    # Handle missing values encoded as -99.99
    co2_col = "Carbon Dioxide (ppm)"
    df[co2_col] = pd.to_numeric(df[co2_col], errors="coerce")
    df = df[df[co2_col] > 0]

    df_annual = df.groupby("Year")[co2_col].mean().reset_index()
    df_annual = df_annual.rename(columns={"Year": "year", co2_col: "co2_ppm"})
    df_annual["year"] = df_annual["year"].astype(int)
    return df_annual


def merge_climate_data(temperature, co2, atmospheric_co2):
    """Inner-join all three datasets on year."""
    merged = pd.merge(temperature, co2, on="year", how="inner")
    merged = pd.merge(merged, atmospheric_co2, on="year", how="inner")
    merged = merged.sort_values("year").reset_index(drop=True)
    return merged


def get_data_quality_report(df):
    """Return a summary dict of data quality metrics."""
    report = {
        "total_rows": len(df),
        "year_range": f"{int(df['year'].min())} – {int(df['year'].max())}",
        "missing_values": int(df.isnull().sum().sum()),
        "duplicate_rows": int(df.duplicated().sum()),
        "columns": list(df.columns),
    }
    return report


if __name__ == "__main__":
    temperature = load_temperature_data("data/raw/temperature.csv")
    co2 = load_co2_data("data/raw/co2_emissions.csv")
    atm_co2 = load_atmospheric_co2_data("data/raw/atmospheric_co2.csv")

    climate = merge_climate_data(temperature, co2, atm_co2)
    climate.to_csv("data/processed/climate_data.csv", index=False)

    report = get_data_quality_report(climate)
    print("Data Quality Report:")
    for k, v in report.items():
        print(f"  {k}: {v}")
    print(climate.head())
