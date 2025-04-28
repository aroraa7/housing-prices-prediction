# Ames Housing Price Prediction

## Overview
This repository implements an end-to-end machine learning pipeline to predict house sale prices on the Ames, Iowa dataset. It covers data cleaning, feature engineering, feature selection, and model benchmarking to identify the most reliable predictor.

## Repository Structure
```
├── basicinfo_housing.py          # Initial data exploration and cleaning script
├── EDA+MainScript-housing.py     # Primary pipeline: preprocessing, modeling, and evaluation
├── housing-price.csv             # Raw Ames housing dataset (1,460 records, 80 features)
└── README.md                     # Project overview and instructions
```

## Setup & Dependencies
1. **Clone** the repo:
   ```bash
   git clone https://github.com/aroraa7/ames-housing.git
   cd ames-housing
   ```
2. **Install** required packages:
   ```bash
   pip install -r requirements.txt
   ```

> _Note: `requirements.txt` should list pandas, numpy, scikit-learn, xgboost, matplotlib, seaborn, scipy._

## Pipeline Steps
1. **Data Cleaning** (`basicinfo_housing.py`):
   - Remove features with >45% missing or zero values.
   - Log-transform skewed numeric columns and handle categorical encodings.
2. **Feature Selection & Modeling** (`EDA+MainScript-housing.py`):
   - Apply LassoCV to select top predictors.
   - Train and compare Linear Regression, SVM, Random Forest, and XGBoost using 10-fold CV.
   - Evaluate models with mean squared error (MSE) and paired t-tests.
3. **Results**:
   - Random Forest delivered the lowest CV MSE and proved most robust against overfitting.
   - Detailed metrics printed to console; plots saved if configured in script.

## How to Run
```bash
python EDA+MainScript-housing.py
```  
*(Ensure `housing-price.csv` is in the same directory.)*

## Key Findings
- **Random Forest** consistently outperforms other models in cross-validated MSE.
- **Lasso**-based feature selection reduces dimensionality from ~748 to a manageable subset.
- **Log transformations** and careful outlier handling improve model stability.


---
_Developed by Ashria Arora — March 2025._

