# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Load the dataset
df = pd.read_csv("housing-price.csv")

num_rows = df.shape[0]
print(f"Total number of rows (observations): {num_rows}")

# Count total number of data points in the dataset
total_data_points = df.shape[0] * df.shape[1]
print(f"Total number of data points in dataset: {total_data_points}")

# Replace "?" with NaN to properly recognize missing values
df.replace("?", np.nan, inplace=True)

# Display basic info
print(df.info())
print(df.head())

# Check for missing values
# Check for missing values
missing_values = df.isnull().sum()

# Exclude "None" in MasVnrType from missing value count
missing_values_corrected = missing_values.drop(labels=["MasVnrType"], errors="ignore")

# Show only columns with missing values
print(missing_values_corrected[missing_values_corrected > 0])

# Sum total missing values in the dataset
total_missing = missing_values.sum()
print(f"Total missing values in dataset: {total_missing}")

missing_percentage = (total_missing / total_data_points) * 100
print(f"Percentage of missing values in dataset: {missing_percentage:.2f}%")

# Basic summary statistics
print(df.describe())

# Visualize SalePrice distribution
sns.histplot(df["SalePrice"], bins=30, kde=True)
plt.title("Distribution of House Prices")
plt.show()



