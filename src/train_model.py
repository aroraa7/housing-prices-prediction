from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVR
from xgboost import XGBRegressor


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = PROJECT_ROOT / Path("data/housing-price.csv")
FIGURES_DIR = PROJECT_ROOT / Path("outputs/figures")
METRICS_DIR = PROJECT_ROOT / Path("outputs/metrics")

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5
TARGET = "SalePrice"


def make_one_hot_encoder():
    """Create an encoder that works across recent scikit-learn versions."""
    try:
        return OneHotEncoder(handle_unknown="ignore", sparse_output=False)
    except TypeError:
        return OneHotEncoder(handle_unknown="ignore", sparse=False)


def load_data(path=DATA_PATH):
    return pd.read_csv(path, na_values="?")


def split_features_target(df):
    if TARGET not in df.columns:
        raise ValueError(f"Expected target column '{TARGET}' in dataset.")

    X = df.drop(columns=[TARGET])
    X = X.drop(columns=["Id"], errors="ignore")
    y = np.log1p(df[TARGET])
    return train_test_split(X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE)


def build_preprocessor(X):
    numeric_features = X.select_dtypes(include=["number"]).columns.tolist()
    categorical_features = X.select_dtypes(exclude=["number"]).columns.tolist()

    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", make_one_hot_encoder()),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, numeric_features),
            ("categorical", categorical_pipeline, categorical_features),
        ]
    )


def build_models(preprocessor):
    return {
        "Linear Regression": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", LinearRegression()),
            ]
        ),
        "SVR": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", SVR()),
            ]
        ),
        "Random Forest": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    RandomForestRegressor(
                        n_estimators=300,
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
        "XGBoost": Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                (
                    "model",
                    XGBRegressor(
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=3,
                        subsample=0.8,
                        colsample_bytree=0.8,
                        objective="reg:squarederror",
                        random_state=RANDOM_STATE,
                        n_jobs=-1,
                    ),
                ),
            ]
        ),
    }


def evaluate_models(models, X_train, X_test, y_train, y_test):
    cv = KFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    results = []
    fitted_models = {}

    for name, model in models.items():
        cv_scores = cross_val_score(
            model,
            X_train,
            y_train,
            scoring="neg_root_mean_squared_error",
            cv=cv,
        )
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        rmse_log = np.sqrt(mean_squared_error(y_test, predictions))
        rmse_dollars = np.sqrt(
            mean_squared_error(np.expm1(y_test), np.expm1(predictions))
        )

        results.append(
            {
                "model": name,
                "cv_rmse_log": -cv_scores.mean(),
                "cv_rmse_log_std": cv_scores.std(),
                "test_rmse_log": rmse_log,
                "test_rmse_dollars": rmse_dollars,
                "test_r2_log": r2_score(y_test, predictions),
            }
        )
        fitted_models[name] = model

    results_df = pd.DataFrame(results).sort_values("cv_rmse_log").reset_index(drop=True)
    return results_df, fitted_models


def save_model_comparison_plot(results_df):
    plt.figure(figsize=(9, 5))
    sns.barplot(data=results_df, x="cv_rmse_log", y="model", color="#4C78A8")
    plt.xlabel("Cross-validated RMSE (log SalePrice)")
    plt.ylabel("")
    plt.title("Model Comparison")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "model_comparison.png", dpi=150)
    plt.close()


def save_prediction_plots(best_model, X_test, y_test, model_name):
    predictions_log = best_model.predict(X_test)
    actual_prices = np.expm1(y_test)
    predicted_prices = np.expm1(predictions_log)
    residuals = actual_prices - predicted_prices

    plt.figure(figsize=(6, 6))
    sns.scatterplot(x=actual_prices, y=predicted_prices, alpha=0.7)
    min_price = min(actual_prices.min(), predicted_prices.min())
    max_price = max(actual_prices.max(), predicted_prices.max())
    plt.plot([min_price, max_price], [min_price, max_price], color="#E45756")
    plt.xlabel("Actual Sale Price")
    plt.ylabel("Predicted Sale Price")
    plt.title(f"Predicted vs Actual: {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "predicted_vs_actual.png", dpi=150)
    plt.close()

    plt.figure(figsize=(8, 5))
    sns.scatterplot(x=predicted_prices, y=residuals, alpha=0.7)
    plt.axhline(0, color="#E45756")
    plt.xlabel("Predicted Sale Price")
    plt.ylabel("Residual")
    plt.title(f"Residuals: {model_name}")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "residuals.png", dpi=150)
    plt.close()


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    X_train, X_test, y_train, y_test = split_features_target(df)
    preprocessor = build_preprocessor(X_train)
    models = build_models(preprocessor)

    results_df, fitted_models = evaluate_models(models, X_train, X_test, y_train, y_test)
    results_df.to_csv(METRICS_DIR / "model_results.csv", index=False)

    best_model_name = results_df.loc[0, "model"]
    best_model = fitted_models[best_model_name]
    save_model_comparison_plot(results_df)
    save_prediction_plots(best_model, X_test, y_test, best_model_name)

    print("Model results saved to outputs/metrics/model_results.csv")
    print(results_df.to_string(index=False))
    print(f"\nBest model by CV RMSE: {best_model_name}")


if __name__ == "__main__":
    main()
