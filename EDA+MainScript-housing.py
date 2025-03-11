# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

### Cleaning the dataset before performing models ###

# Load the dataset
df = pd.read_csv("housing-price.csv")

# Define a threshold (e.g., 50% missing values) and missing_values
threshold = 0.5  # 50%

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
print(f"Dropped columns: {cols_to_drop}")

# Check the new shape of the dataset
print(f"New dataset shape: {df.shape}")