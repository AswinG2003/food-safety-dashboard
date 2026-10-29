# Chicago Food Safety Risk Dashboard

A Streamlit dashboard for visualizing estimated restaurant inspection failure risks across Chicago, powered by standalone Random Forest and XGBoost models.

## Quick Start

```bash
# From the 03_working_app directory:
pip install -r requirements.txt
streamlit run app.py
```

## Dashboard Sections

| Tab | Description |
|-----|-------------|
| **📊 Overview** | Citywide KPIs, risk distribution histogram, priority tier breakdown, model selector |
| **🗺️ Risk Map** | Interactive Chicago map with risk-coloured markers, filterable by risk category and probability |
| **📋 Inspector Queue** | Ranked restaurant table by estimated failure probability, with filters and CSV download |
| **🏪 Restaurant Report Card** | Look up any restaurant by ID; see risk scores, inspection history, violation excerpts |
| **🔬 What-If Simulator** | Explore how violation count, inspection type, age, and risk category relate to predicted risk |
| **⚠️ Violation Analytics** | Most common violation numbers, violation count distributions, risk-by-violation scatter |
| **📅 Seasonality & Geography** | Inspection volume trends, monthly failure rates, geographic grid risk comparison |
| **🧠 Model Insights** | PR-AUC, ROC-AUC, Brier score, top-k failure rates, calibration curves, feature importance |

## Data Sources

All data lives in `../02_my_model_artifacts/data/`:

| File | Purpose |
|------|---------|
| `restaurant_inspection_priority_list.csv` | Main restaurant data with precomputed probabilities for all models |
| `model_metrics_all.csv` | Test-set metrics for all models (dashboard uses only RF and XGBoost rows) |
| `random_forest_feature_importance.csv` | RF model-native feature importances (Gini impurity) |
| `xgboost_feature_importance.csv` | XGBoost model-native feature importances (gain) |
| `calibration_curve_test.csv` | Calibration curve data for reliability diagrams |
| `threshold_metrics.csv` | Confusion matrix metrics at various decision thresholds |
| `test_predictions_ranked.csv` | Individual test-set predictions with actual outcomes |
| `predict_row_metadata.csv` | Inspection history metadata for restaurant lookups |

## Model Selection

The sidebar model selector switches between **Random Forest** and **XGBoost** standalone models. Changing the model:
- Updates all probability scores, rankings, and priority tiers
- Recalculates the Inspector Queue ordering
- Updates the Risk Map marker colours
- Changes the highlighted metrics in Model Insights

## Important Notes

- **Estimated risks, not official classifications.** Probabilities are sigmoid-calibrated model estimates. They are decision-support signals, not regulatory determinations.
- **Priority tiers are operational buckets.** P1–P5 tiers are percentile-based rankings computed from the selected model's probabilities.
- **No live inference.** The models require 56 encoded features including cumulative inspection history. The full preprocessing pipeline is not bundled, so the dashboard uses precomputed probabilities.
- **Feature importance is model-native, not SHAP.** Displayed importances use Gini impurity (RF) or gain (XGBoost).
- **No external API keys required.**

## Requirements

- Python 3.9+
- Streamlit 1.28+
- pandas, numpy, plotly, pydeck
