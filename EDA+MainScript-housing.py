# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

### Cleaning the dataset before performing models ###

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


### Creating Correlation Matrix ###


import matplotlib.pyplot as plt
import seaborn as sns

# Select only numerical columns for correlation calculation
numerical_df = df.select_dtypes(include=['int64', 'float64'])

# Compute the correlation matrix
correlation_matrix = numerical_df.corr()

# Plot the heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, cmap="coolwarm", annot=False, fmt=".2f", linewidths=0.5)
plt.title("Correlation Matrix (Only Numerical Features)")
plt.show()


