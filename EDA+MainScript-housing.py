# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

#################### Cleaning Dataset ####################

# Load the dataset
df = pd.read_csv("housing-price.csv")

# Define a threshold and missing_values
threshold = 0.45

# Replace "?" with NaN to properly recognize missing values
df.replace("?", np.nan, inplace=True)
missing_values = df.isnull().sum()

# Exclude "None" in MasVnrType from missing value count
missing_values_corrected = missing_values.drop(labels=["MasVnrType"], errors="ignore")


# Identify columns where the percentage of missing values is greater than the threshold
cols_to_drop = missing_values_corrected[missing_values_corrected > threshold * df.shape[0]].index.tolist()

# Drop these columns
df.drop(columns=cols_to_drop, inplace=True)

# Print the columns that were dropped
# print(f"Dropped columns: {cols_to_drop}")

# Check the new shape of the dataset
# print(f"New dataset shape: {df.shape}")

missing_values_new = df.isnull().sum()

# Exclude "None" in MasVnrType from missing value count
missing_values_corrected_new = missing_values_new.drop(labels=["MasVnrType"], errors="ignore")
# print(missing_values_corrected_new[missing_values_corrected_new > 0])

## checking to see if more columns need to be dropped
# Select only numerical columns
numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns

# Count zeros in each numerical column
zero_counts = (df[numeric_cols] == 0).sum()

## Checking to see if more columns need to be dropped due to excessive zeros ##
zero_threshold = 0.45  # 45% of dataset

# Identify columns where the percentage of zeros is greater than the threshold
cols_to_drop_zeros = zero_counts[zero_counts > zero_threshold * df.shape[0]].index.tolist()

# Drop these columns
df.drop(columns=cols_to_drop_zeros, inplace=True)

# Print dropped columns due to excessive zeros
# print(f"Dropped columns due to excessive zeros: {cols_to_drop_zeros}")

# Check new shape of the dataset after dropping zero-heavy columns
# print(f"New dataset shape after dropping zero-heavy columns: {df.shape}")

# Print only columns where zeros are present (after removing excessive zero columns)
# Recalculate zero counts after dropping columns
numeric_cols_updated = df.select_dtypes(include=['int64', 'float64']).columns  # Updated numeric columns
zero_counts_filtered = (df[numeric_cols_updated] == 0).sum()

# Print only columns where zeros are still present
# print(zero_counts_filtered[zero_counts_filtered > 0])


#################### Creating Correlation Matrix ####################


# Select only numerical columns for correlation calculation
numerical_df = df.select_dtypes(include=['int64', 'float64'])

# Compute the correlation matrix
correlation_matrix = numerical_df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, fmt=".2f", linewidths=0.5)
# plt.title("Correlation Matrix (Only Numerical Features)")
# plt.show()



#################### Outlier Detection ####################

## Exclude numerical features that are actually categorical or ordinal
categorical_numerical_features = ['MSSubClass', 'OverallQual', 'OverallCond', 'MoSold', 'YrSold', 'GarageCars']

# Select only continuous numerical features, excluding the target variable and categorical-like ones
numerical_features = df.select_dtypes(include=['int64', 'float64']).drop(columns=['SalePrice'] + categorical_numerical_features, errors='ignore')

### Detect Outliers in Continuous Numerical Features using IQR ###
def detect_outliers_iqr(df, features):
    outlier_counts = {}
    for feature in features:
        Q1 = df[feature].quantile(0.25)
        Q3 = df[feature].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        outliers = df[(df[feature] < lower_bound) | (df[feature] > upper_bound)]
        outlier_counts[feature] = len(outliers)
    
    return pd.DataFrame(outlier_counts.items(), columns=['Feature', 'Outlier Count']).sort_values(by='Outlier Count', ascending=False)

# Get outlier counts for continuous numerical features
outlier_results = detect_outliers_iqr(df, numerical_features.columns)
# print(outlier_results)

### Visualizing Outliers for Continuous Numerical Features ###
# Adjust the number of rows and columns dynamically based on the number of continuous numerical features
num_features = len(numerical_features.columns)
num_cols = 5  # Number of columns for subplots
num_rows = (num_features // num_cols) + (num_features % num_cols > 0)  # Calculate required rows

# Create boxplots for all continuous numerical features
plt.figure(figsize=(20, num_rows * 4))
for i, feature in enumerate(numerical_features.columns, 1):
    plt.subplot(num_rows, num_cols, i)
    sns.boxplot(y=df[feature])
    plt.title(feature)

# plt.tight_layout()
# plt.show()



#################### Dummy Vairables for Categorical Features ####################

# Identify categorical columns
categorical_features = df.select_dtypes(include=['object']).columns.tolist()

# Apply get_dummies to encode categorical features
df = pd.get_dummies(df, columns=categorical_features, drop_first=True)

# Convert boolean values to integers (0s and 1s)
df = df.astype(int)

# print(df)

# print("Dummy variables created for categorical features.")
print(f"New dataset shape: {df.shape}")


#################### Log Transform Continuous Features ####################
# print(df)
# Select only continuous numerical features for log transformation
numerical_features = df.select_dtypes(include=['int64', 'float64']).drop(columns=['SalePrice'] + categorical_numerical_features, errors='ignore')

# Apply log transformation using np.log1p (log(1 + x) to avoid log(0) issues)
df[numerical_features.columns] = df[numerical_features.columns].apply(lambda x: np.log1p(x))

# Apply log transformation to the target variable (SalePrice)
df['SalePrice'] = np.log1p(df['SalePrice'])

# print("Log transformation applied to continuous numerical features and target variable.")

# print(df)



#################### Split Data Train/Test sets ####################


from sklearn.model_selection import train_test_split

# Separate target variable (SalePrice) and features
X = df.drop(columns=['SalePrice'])  # Features
y = df['SalePrice']  # Target variable

# Split data into training (80%) and testing (20%)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# print(f"Training set shape: X_train: {X_train.shape}, y_train: {y_train.shape}")
# print(f"Test set shape: X_test: {X_test.shape}, y_test: {y_test.shape}")


#################### Perform Lasso to Select Features ####################

from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Standardize numerical features (Lasso is sensitive to scale)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train Lasso with Cross-Validation to find best alpha
lasso = LassoCV(cv=5, random_state=42).fit(X_train_scaled, y_train)

# Get feature importance (nonzero coefficients)
selected_features = np.array(X_train.columns)[lasso.coef_ != 0]

# Keep only selected features
X_train_selected = X_train[selected_features]
X_test_selected = X_test[selected_features]
# print(selected_features)

# print(f"Number of selected features: {len(selected_features)}")
# print(f"Reduced training set shape: {X_train_reduced.shape}")
# print(f"Reduced test set shape: {X_test_reduced.shape}")






#################### Perform Baseline models, random forest, and XGBoost without and with CV ####################



# from sklearn.model_selection import cross_val_score
# from sklearn.linear_model import LinearRegression
# from sklearn.svm import SVR
# from sklearn.metrics import mean_squared_error, r2_score

# # Initialize models
# lin_reg = LinearRegression()
# svm_reg = SVR(kernel='rbf')  # RBF kernel for non-linear relationships

# # Perform Cross-Validation on Training Data (MSE)
# lin_reg_cv_mse = cross_val_score(lin_reg, X_train_reduced, y_train, 
#                                  scoring="neg_mean_squared_error", cv=5)

# svm_cv_mse = cross_val_score(svm_reg, X_train_reduced, y_train, 
#                              scoring="neg_mean_squared_error", cv=5)

# # Convert scores to positive MSE
# lin_reg_mse = -lin_reg_cv_mse.mean()
# svm_mse = -svm_cv_mse.mean()

# # Perform Cross-Validation on Training Data (R² Score)
# lin_reg_cv_r2 = cross_val_score(lin_reg, X_train_reduced, y_train, 
#                                 scoring="r2", cv=5)

# svm_cv_r2 = cross_val_score(svm_reg, X_train_reduced, y_train, 
#                             scoring="r2", cv=5)

# # Compute mean R² scores
# lin_reg_r2 = lin_reg_cv_r2.mean()
# svm_r2 = svm_cv_r2.mean()

# print(f"Cross-Validation Results (Training Set)")
# print(f"   - Linear Regression MSE: {lin_reg_mse:.4f}, R²: {lin_reg_r2:.4f}")
# print(f"   - SVM Regression MSE: {svm_mse:.4f}, R²: {svm_r2:.4f}")

# # Train the models on the full training set
# lin_reg.fit(X_train_reduced, y_train)
# svm_reg.fit(X_train_reduced, y_train)

# # Evaluate on the test set
# y_pred_lin = lin_reg.predict(X_test_reduced)
# y_pred_svm = svm_reg.predict(X_test_reduced)

# # Compute test set performance
# lin_reg_test_mse = mean_squared_error(y_test, y_pred_lin)
# svm_test_mse = mean_squared_error(y_test, y_pred_svm)

# lin_reg_test_r2 = r2_score(y_test, y_pred_lin)
# svm_test_r2 = r2_score(y_test, y_pred_svm)

# print(f"\n Final Model Performance on Test Set")
# print(f"   - Linear Regression MSE: {lin_reg_test_mse:.4f}, R²: {lin_reg_test_r2:.4f}")
# print(f"   - SVM Regression MSE: {svm_test_mse:.4f}, R²: {svm_test_r2:.4f}")


# #################### Perform Random Forest and XGboost selected features from Lasso ####################
# from sklearn.ensemble import RandomForestRegressor
# # from xgboost import XGBRegressor
# from sklearn.metrics import mean_squared_error, r2_score

# # Keep only Lasso-selected features
# X_train_reduced = X_train[selected_features]
# X_test_reduced = X_test[selected_features]

# # Initialize models
# rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
# # xgb_model = XGBRegressor(n_estimators=100, random_state=42)

# # Train models
# rf_model.fit(X_train_reduced, y_train)
# # xgb_model.fit(X_train_reduced, y_train)

# # Make predictions
# y_pred_rf = rf_model.predict(X_test_reduced)
# # y_pred_xgb = xgb_model.predict(X_test_reduced)

# # Compute performance metrics
# rf_mse = mean_squared_error(y_test, y_pred_rf)
# rf_r2 = r2_score(y_test, y_pred_rf)

# # xgb_mse = mean_squared_error(y_test, y_pred_xgb)
# # xgb_r2 = r2_score(y_test, y_pred_xgb)

# # Print results
# print(f"Random Forest Performance:")
# print(f"   - Mean Squared Error (MSE): {rf_mse:.4f}")
# print(f"   - R² Score: {rf_r2:.4f}\n")

# # print(f"XGBoost Performance:")
# # print(f"   - Mean Squared Error (MSE): {xgb_mse:.4f}")
# # print(f"   - R² Score: {xgb_r2:.4f}")


from sklearn.model_selection import cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor
# import xgboost as XGBRegressor
from sklearn.metrics import mean_squared_error, r2_score


### Linear Regression ###
lin_reg = LinearRegression()
lin_reg.fit(X_train_selected, y_train)

# Training and testing predictions
y_train_pred = lin_reg.predict(X_train_selected)
y_test_pred = lin_reg.predict(X_test_selected)

# Compute errors
lin_reg_train_mse = mean_squared_error(y_train, y_train_pred)
lin_reg_test_mse = mean_squared_error(y_test, y_test_pred)
lin_reg_train_r2 = r2_score(y_train, y_train_pred)
lin_reg_test_r2 = r2_score(y_test, y_test_pred)

print(f"Linear Regression Training Error: {lin_reg_train_mse:.4f}, R²: {lin_reg_train_r2:.4f}")
print(f"Linear Regression Testing Error: {lin_reg_test_mse:.4f}, R²: {lin_reg_test_r2:.4f}")

# Cross-validation
lin_reg_cv_mse = -np.mean(cross_val_score(lin_reg, X_train_selected, y_train, scoring='neg_mean_squared_error', cv=10))
print(f"Linear Regression Cross-Validation MSE: {lin_reg_cv_mse:.4f}")

### Support Vector Machine ###
svm_reg = SVR()
svm_reg.fit(X_train_selected, y_train)

y_train_pred = svm_reg.predict(X_train_selected)
y_test_pred = svm_reg.predict(X_test_selected)

svm_train_mse = mean_squared_error(y_train, y_train_pred)
svm_test_mse = mean_squared_error(y_test, y_test_pred)
svm_train_r2 = r2_score(y_train, y_train_pred)
svm_test_r2 = r2_score(y_test, y_test_pred)

print(f"SVM Training Error: {svm_train_mse:.4f}, R²: {svm_train_r2:.4f}")
print(f"SVM Testing Error: {svm_test_mse:.4f}, R²: {svm_test_r2:.4f}")

# Cross-validation
svm_cv_mse = -np.mean(cross_val_score(svm_reg, X_train_selected, y_train, scoring='neg_mean_squared_error', cv=10))
print(f"SVM Cross-Validation MSE: {svm_cv_mse:.4f}")

### Random Forest ###
rf_reg = RandomForestRegressor(n_estimators=100, random_state=42)
rf_reg.fit(X_train_selected, y_train)

y_train_pred = rf_reg.predict(X_train_selected)
y_test_pred = rf_reg.predict(X_test_selected)

rf_train_mse = mean_squared_error(y_train, y_train_pred)
rf_test_mse = mean_squared_error(y_test, y_test_pred)
rf_train_r2 = r2_score(y_train, y_train_pred)
rf_test_r2 = r2_score(y_test, y_test_pred)

print(f"Random Forest Training Error: {rf_train_mse:.4f}, R²: {rf_train_r2:.4f}")
print(f"Random Forest Testing Error: {rf_test_mse:.4f}, R²: {rf_test_r2:.4f}")

# Cross-validation
rf_cv_mse = -np.mean(cross_val_score(rf_reg, X_train_selected, y_train, scoring='neg_mean_squared_error', cv=10))
print(f"Random Forest Cross-Validation MSE: {rf_cv_mse:.4f}")

## XGBoost ###
# xgb_reg = XGBRegressor(n_estimators=100, random_state=42, n_jobs = 1)
# xgb_reg.fit(X_train_selected, y_train)

# y_train_pred = xgb_reg.predict(X_train_selected)
# y_test_pred = xgb_reg.predict(X_test_selected)

# xgb_train_mse = mean_squared_error(y_train, y_train_pred)
# xgb_test_mse = mean_squared_error(y_test, y_test_pred)
# xgb_train_r2 = r2_score(y_train, y_train_pred)
# xgb_test_r2 = r2_score(y_test, y_test_pred)

# print(f"XGBoost Training Error: {xgb_train_mse:.4f}, R²: {xgb_train_r2:.4f}")
# print(f"XGBoost Testing Error: {xgb_test_mse:.4f}, R²: {xgb_test_r2:.4f}")

# # Cross-validation
# xgb_cv_mse = -np.mean(cross_val_score(xgb_reg, X_train_selected, y_train, scoring='neg_mean_squared_error', cv=5))
# print(f"XGBoost Cross-Validation MSE: {xgb_cv_mse:.4f}")
