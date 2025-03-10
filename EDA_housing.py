# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Load the dataset
df = pd.read_csv("housing-price.csv")

# Display basic info
print(df.info())
print(df.head())

# Check for missing values
print(df.isnull().sum())

# Basic summary statistics
print(df.describe())

# Visualize SalePrice distribution
sns.histplot(df["SalePrice"], bins=30, kde=True)
plt.title("Distribution of House Prices")
plt.show()