# Ames Housing Price Prediction

## Objective
Predict residential sale prices using structured property features from the Ames, Iowa housing dataset. The project frames house price prediction as a practical real estate valuation problem where accurate estimates can support pricing strategy, market analysis, and property comparison.

## Dataset
The dataset contains 1,460 home sales with 80 original feature columns and the target variable `SalePrice`. Features describe property quality, lot characteristics, living area, basement and garage attributes, neighborhood, sale conditions, and other housing details.

The raw CSV is stored at `data/housing-price.csv`.

## Modeling Approach
The cleaned modeling workflow is implemented in `src/train_model.py`.

Key steps:

- Load the raw dataset with `?` values treated as missing.
- Drop the row identifier column from model features.
- Split data into training and test sets before fitting preprocessing steps.
- Log-transform `SalePrice` with `np.log1p` to reduce target skew.
- Use sklearn pipelines for reproducible preprocessing:
  - numeric features: median imputation and standard scaling
  - categorical features: most-frequent imputation and one-hot encoding
- Benchmark Linear Regression, SVR, Random Forest, and XGBoost.
- Compare models using cross-validated RMSE, test RMSE, and test R2.

The cleaned pipeline is the main portfolio entrypoint.

## Results
Running the training script writes model metrics to:

```text
outputs/metrics/model_results.csv
```

The results file includes:

| Metric | Description |
| --- | --- |
| `cv_rmse_log` | Mean cross-validated RMSE on log-transformed sale price |
| `cv_rmse_log_std` | Standard deviation of CV RMSE across folds |
| `test_rmse_log` | Holdout test RMSE on log-transformed sale price |
| `test_rmse_dollars` | Approximate holdout RMSE after converting predictions back to dollars |
| `test_r2_log` | Holdout R2 on log-transformed sale price |

The script also saves portfolio-ready figures to `outputs/figures/`:

- model comparison chart
- predicted vs actual sale prices
- residual plot

## Key Insights
Expected price drivers in this dataset include overall home quality, above-ground living area, neighborhood, year built or remodeled, basement size, and garage capacity. The cleaned workflow is designed to make those relationships easier to evaluate through model performance and saved diagnostic plots.

## Repository Structure
```text
├── data/
│   └── housing-price.csv
├── outputs/
│   ├── figures/
│   └── metrics/
├── src/
│   └── train_model.py
├── README.md
├── requirements.txt
└── .gitignore
```

## Setup
Create and activate a virtual environment, then install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run
From the project root:

```bash
python3 src/train_model.py
```

## Next Steps
- Add hyperparameter tuning for the strongest models.
- Add SHAP or permutation importance for model explainability.
- Create a polished EDA notebook for visual storytelling.
- Package the final model behind a small API or Streamlit app.

---

Developed by Ashria Arora.
