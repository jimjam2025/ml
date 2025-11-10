import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Assignment 2: Implementation of Pandas and matplotlib

# a) Create a Series using pandas and display
print("--- Assignment 2a ---")
s = pd.Series([1, 3, 5, np.nan, 6, 8])
print(f"Series:\n{s}")
print("\n")

# b) Access the index and the values of our Series
print("--- Assignment 2b ---")
print(f"Index: {s.index}")
print(f"Values: {s.values}")
print("\n")

# c) Compare an array using Numpy with a series using pandas
print("--- Assignment 2c ---")
arr = np.array([1, 3, 5, 7, 6, 8])
print(f"Numpy array: {arr}")
# Creating a series from the numpy array for comparison
series_from_arr = pd.Series(arr)
print(f"Pandas series from array:\n{series_from_arr}")
# Comparing the original series (with NaN dropped) with the numpy array
print(f"Original Pandas series (NaN dropped): \n{s[s.notna()].astype(int)}")
print("\n")

# d) Define Series objects with individual indices
print("--- Assignment 2d ---")
s_indexed = pd.Series([1, 2, 3], index=['a', 'b', 'c'])
print(f"Series with index:\n{s_indexed}")
print("\n")

# e) Access single value of a series
print("--- Assignment 2e ---")
print(f"Value at index 'b': {s_indexed['b']}")
print("\n")

# f) Load datasets in a Data frame variable using pandas
print("--- Assignment 2f ---")
# Creating a dictionary to be loaded into a DataFrame
data = {'Name': ['Alice', 'Bob', 'Charlie'], 'Age': [25, 30, 22], 'City': ['New York', 'London', 'Paris']}
df = pd.DataFrame(data)
print(f"DataFrame:\n{df}")
print("\n")

# Usage of different methods in Matplotlib.
print("--- Matplotlib Example ---")

# Data for plotting
x = np.linspace(0, 10, 100)
y_sin = np.sin(x)
y_cos = np.cos(x)

# Create a plot
plt.figure(figsize=(8, 6))

# Plotting sine and cosine waves
plt.plot(x, y_sin, label='sin(x)', color='blue', linestyle='-')
plt.plot(x, y_cos, label='cos(x)', color='red', linestyle='--')

# Adding title and labels
plt.title("Sine and Cosine Waves")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")

# Adding a legend
plt.legend()

# Adding a grid
plt.grid(True)

# Display the plot
plt.show()

# Another example: Scatter plot
print("--- Matplotlib Scatter Plot Example ---")
# Data for scatter plot
x_scatter = np.random.rand(50)
y_scatter = np.random.rand(50)
colors = np.random.rand(50)
sizes = 1000 * np.random.rand(50)

# Create a scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(x_scatter, y_scatter, c=colors, s=sizes, alpha=0.5, cmap='viridis')

# Adding a color bar
plt.colorbar()

# Adding title and labels
plt.title("Scatter Plot Example")
plt.xlabel("X Value")
plt.ylabel("Y Value")

# Display the plot
plt.show()
