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
# print(f"New dataset shape: {df.shape}")


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
X_train_reduced = X_train[selected_features]
X_test_reduced = X_test[selected_features]
print(selected_features)
print(f"Number of selected features: {len(selected_features)}")
print(f"Reduced training set shape: {X_train_reduced.shape}")
print(f"Reduced test set shape: {X_test_reduced.shape}")



