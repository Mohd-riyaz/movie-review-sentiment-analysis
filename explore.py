import pandas as pd

# Load the dataset
df = pd.read_csv("data/IMDB_Dataset.csv")  # Change the filename if yours is different

# First 5 rows
print("=" * 50)
print("First 5 Rows")
print("=" * 50)
print(df.head())

# Shape of the dataset
print("\n" + "=" * 50)
print("Dataset Shape")
print("=" * 50)
print(df.shape)

# Column names
print("\n" + "=" * 50)
print("Column Names")
print("=" * 50)
print(df.columns.tolist())

# Data types
print("\n" + "=" * 50)
print("Data Types")
print("=" * 50)
print(df.dtypes)

# Missing values
print("\n" + "=" * 50)
print("Missing Values")
print("=" * 50)
print(df.isnull().sum())

# Duplicate rows
print("\n" + "=" * 50)
print("Duplicate Rows")
print("=" * 50)
print(df.duplicated().sum())

# Sentiment counts
print("\n" + "=" * 50)
print("Sentiment Distribution")
print("=" * 50)
print(df["sentiment"].value_counts())

# Dataset information
print("\n" + "=" * 50)
print("Dataset Information")
print("=" * 50)
df.info()

# Basic statistics
print("\n" + "=" * 50)
print("Basic Statistics")
print("=" * 50)
print(df.describe(include="all"))