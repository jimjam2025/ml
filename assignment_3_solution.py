import pandas as pd
import numpy as np
from sklearn.datasets import load_iris, load_digits, load_diabetes
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# Assignment 3: Creation and loading different types of datasets

# Data set Creation:
print("--- Data set Creation ---")

# i. Creation using pandas
print("--- i. Creation using pandas ---")
# 1. From a Dictionary:
data_dict = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 22], 'City': ['New York', 'London', 'Paris']}
df_from_dict = pd.DataFrame(data_dict)
print("1. DataFrame from Dictionary:")
print(df_from_dict)
print("\n")

# 2. From a List of Lists:
data_list_of_lists = [['Alice', 25, 'New York'], ['Bob', 30, 'London'], ['Charlie', 22, 'Paris']]
df_from_list_of_lists = pd.DataFrame(data_list_of_lists, columns=['Name', 'Age', 'City'])
print("2. DataFrame from List of Lists:")
print(df_from_list_of_lists)
print("\n")

# 3. From a List of Dictionaries:
data_list_of_dicts = [{'Name': 'Alice', 'Age': 25, 'City': 'New York'}, {'Name': 'Bob', 'Age': 30, 'City': 'London'}, {'Name': 'Charlie', 'Age': 22, 'City': 'Paris'}]
df_from_list_of_dicts = pd.DataFrame(data_list_of_dicts)
print("3. DataFrame from List of Dictionaries:")
print(df_from_list_of_dicts)
print("\n")

# 4. From External Files (CSV, Excel, etc.):
# Create a dummy CSV file to read from
df_from_dict.to_csv('data.csv', index=False)
df_csv = pd.read_csv('data.csv')
print("4. DataFrame from CSV file:")
print(df_csv.head())
print("\n")

# 5. From a NumPy Array:
data_np = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
df_np = pd.DataFrame(data_np, columns=['ColA', 'ColB', 'ColC'])
print("5. DataFrame from NumPy Array:")
print(df_np)
print("\n")

# ii. Loading CSV dataset files using Pandas is shown above.

# iii. Loading datasets using sklearn
print("--- iii. Loading datasets using sklearn ---")
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
print("Iris dataset (first 5 rows):\n", X_iris[:5])
print("\n")

digits = load_digits()
X_digits, y_digits = digits.data, digits.target
print("Digits dataset (first 5 rows):\n", X_digits[:5])
print("\n")

diabetes = load_diabetes()
X_diabetes, y_diabetes = diabetes.data, diabetes.target
print("Diabetes dataset (first 5 rows):\n", X_diabetes[:5])
print("\n")


# iv. Loading data sets into Google Colab
print("--- iv. Loading data sets into Google Colab ---")
print("The following code is for Google Colab and won't run here.")
# from google.colab import files
# uploaded = files.upload()
print("\n")


# b) Write a python program to compute Mean, Median, Mode, Variance, Standard Deviation using Datasets
print("--- Assignment 3b ---")
sample_data = {'Score': [8, 7, 7, 9, 8, 6, 7, 8, 9, 10, 7, 8, 6, 7, 8]}
df_stats = pd.DataFrame(sample_data)
print("Sample Data for Stats:")
print(df_stats)
print(f"Mean: {df_stats['Score'].mean()}")
print(f"Median: {df_stats['Score'].median()}")
print(f"Mode: {df_stats['Score'].mode().to_list()}")
print(f"Variance: {df_stats['Score'].var()}")
print(f"Standard Deviation: {df_stats['Score'].std()}")
print("\n")

# c) Demonstrate various data pre-processing techniques for a given dataset.
print("--- Assignment 3c ---")
# Create a sample dataframe for pre-processing
data_preprocess = {
    'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
    'B': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
    'C': [5, 4, 3, 2, 1, 5, 4, 3, 2, np.nan] # Add a missing value
}
df_preprocess = pd.DataFrame(data_preprocess)
print("Original DataFrame for Pre-processing:")
print(df_preprocess)
print("\n")

# i. Reshaping the data (Melting)
print("--- i. Reshaping the data (Melting) ---")
df_melted = df_preprocess.melt(id_vars=['A'], value_vars=['B', 'C'])
print("Melted DataFrame:")
print(df_melted.head())
print("\n")

# ii. Filtering the data
print("--- ii. Filtering the data ---")
filtered_df = df_preprocess[df_preprocess['B'] > 50]
print("Filtered DataFrame (B > 50):")
print(filtered_df)
print("\n")

# iii. Merging the data
print("--- iii. Merging the data ---")
data_to_merge = {
    'A': [1, 2, 11, 12],
    'D': ['a', 'b', 'c', 'd']
}
df_to_merge = pd.DataFrame(data_to_merge)
merged_df = pd.merge(df_preprocess, df_to_merge, on='A', how='inner') # Inner join
print("Merged DataFrame (inner join on A):")
print(merged_df)
print("\n")


# iv. Handling the missing values in datasets
print("--- iv. Handling the missing values in datasets ---")
# Fill missing values with the mean of the column
df_filled = df_preprocess.copy()
df_filled['C'].fillna(df_filled['C'].mean(), inplace=True)
print("DataFrame with missing values in 'C' filled with mean:")
print(df_filled)
# Or drop rows with missing values
df_dropped = df_preprocess.dropna()
print("\nDataFrame with rows containing missing values dropped:")
print(df_dropped)
print("\n")

# v. Feature Normalization
print("--- v. Feature Normalization ---")

# Min-max normalization
print("--- Min-max normalization ---")
min_max_scaler = MinMaxScaler()
df_min_max_scaled = df_preprocess.copy()
df_min_max_scaled.dropna(inplace=True) # Scaler cannot handle NaNs
df_min_max_scaled[['A', 'B']] = min_max_scaler.fit_transform(df_min_max_scaled[['A', 'B']])
print("Min-Max Scaled DataFrame:")
print(df_min_max_scaled)
print("\n")

# Scalar Normalization (Standardization)
print("--- StandardScaler ---")
standard_scaler = StandardScaler()
df_standard_scaled = df_preprocess.copy()
df_standard_scaled.dropna(inplace=True) # Scaler cannot handle NaNs
df_standard_scaled[['A', 'B']] = standard_scaler.fit_transform(df_standard_scaled[['A', 'B']])
print("Standard Scaled DataFrame:")
print(df_standard_scaled)
