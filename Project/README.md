# 🌍 Climate Change Trend Analysis Platform

A comprehensive data science pipeline for analyzing historical climate data, modeling the relationship between CO₂ emissions and global temperature anomalies, and simulating future climate scenarios under different emission pathways.

## 🎯 What This Project Answers

1. **How have temperature and CO₂ changed historically?**
2. **What statistical relationship exists between them?**
3. **What happens if current trends continue?**
4. **How do predictions change under different future CO₂ scenarios?**

## 🏗️ Architecture

```
Public Climate Datasets → Data Cleaning & Integration → EDA → Feature Engineering
→ Historical Trend Analysis / Correlation / Anomaly Detection / Forecasting Models
→ Model Evaluation → Future Scenario Engine → Interactive Dashboard → Climate Insights
```

## 📁 Project Structure

```
climate-trend-analysis/
├── data/
│   ├── raw/               # Original downloaded datasets
│   └── processed/         # Cleaned, merged dataset
├── src/
│   ├── preprocessing.py   # Data loading, cleaning, merging
│   ├── features.py        # Feature engineering
│   ├── models.py          # Model training, evaluation, feature importance
│   ├── scenarios.py       # Future CO₂ scenario engine
│   └── evaluation.py      # Metrics (MAE, RMSE, R², MAPE)
├── models/                # Saved trained models (.pkl)
├── app/
│   ├── app.py             # Streamlit main page
│   └── pages/             # Multi-page dashboard
│       ├── 1_Historical_Trends.py
│       ├── 2_CO2_Analysis.py
│       ├── 3_Model_Comparison.py
│       ├── 4_Future_Forecast.py
│       ├── 5_Scenario_Simulator.py
│       └── 6_Data_Explorer.py
├── notebooks/             # Jupyter notebooks for experimentation
├── tests/
├── requirements.txt
└── README.md
```

## 📊 Datasets

| Dataset | Source | Coverage |
|---------|--------|----------|
| Global Temperature Anomaly | HadCRUT (Kaggle) | 1900–2021 |
| CO₂ Emissions | Our World in Data | 1750–2023 |
| Atmospheric CO₂ | Mauna Loa / UCSD (Kaggle) | 1958–2017 |

## 🧪 Models

Four baseline models evaluated with **TimeSeriesSplit** (expanding window) validation:

| Model | Description |
|-------|-------------|
| Linear Regression | Simple baseline |
| Random Forest | Ensemble of decision trees |
| XGBoost | Gradient-boosted trees |
| ARIMA(X) | Classical time-series with exogenous variables |

## 🚀 Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Preprocess data (if not already done)
python src/preprocessing.py

# Train models
python src/models.py

# Launch dashboard
streamlit run app/app.py
```

## 🖥️ Dashboard Pages

1. **Home** — Platform overview with key metrics
2. **Historical Trends** — Temperature and CO₂ trends with rolling averages and decade analysis
3. **CO₂ Analysis** — Correlation analysis, scatter plots, country-level emissions
4. **Model Comparison** — Leaderboard, Actual vs Predicted, feature importance
5. **Future Forecast** — Baseline projection with all scenario overlays
6. **Scenario Simulator** — Interactive what-if analysis with custom emission sliders
7. **Data Explorer** — Data quality, anomaly detection, distributions, download

## ⚠️ Important Notes

- **Correlation ≠ Causation**: The statistical relationships shown do not by themselves prove the causal mechanism of climate change.
- **Model Limitations**: These models are trained on ~60 annual data points. They capture trends but are not a substitute for physics-based climate models (e.g., CMIP6).
- **Scenario Assumptions**: The emission pathways are simplified compound-growth assumptions, not policy-calibrated SSP scenarios.

## 🛠️ Tech Stack

- **Data**: pandas, numpy
- **Visualization**: plotly, matplotlib
- **ML**: scikit-learn, xgboost, statsmodels
- **Explainability**: shap
- **Dashboard**: streamlit
