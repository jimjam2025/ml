import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# Assignment 8: k-Nearest Neighbor

# 1. Predict Sugar of Diabetic Patient given BMI and Age using k-NN, assume k = 3
print("--- Assignment 8.1: Predict Sugar of Diabetic Patient ---")
# Data
# Note: The 'O' in the Sugar column is interpreted as 0.
X_diabetes = np.array([
    [33.6, 50],
    [26.6, 30],
    [23.4, 40],
    [43.1, 67],
    [35.3, 23],
    [35.9, 67],
    [36.7, 45],
    [25.7, 46],
    [23.3, 29],
    [31, 56]
])
y_diabetes = np.array([1, 0, 0, 0, 1, 1, 1, 0, 0, 1])

# k-NN model
knn_diabetes = KNeighborsClassifier(n_neighbors=3)
knn_diabetes.fit(X_diabetes, y_diabetes)

# Predict for a new patient (e.g., BMI=30, Age=60)
new_patient = np.array([[30, 60]])
prediction_diabetes = knn_diabetes.predict(new_patient)

print(f"The predicted sugar level for a patient with BMI={new_patient[0][0]} and Age={new_patient[0][1]} is: {prediction_diabetes[0]}")
print("\n")


# 2. Design a k-NN to assign a class label for a new data point
print("--- Assignment 8.2: Assign Class Label using k-NN ---")
# Data
X_color = np.array([
    [40, 20],
    [50, 50],
    [60, 90],
    [10, 25],
    [70, 70],
    [60, 10],
    [25, 80]
])
y_color = np.array(['Red', 'Blue', 'Blue', 'Red', 'Blue', 'Red', 'Blue'])

# k-NN model
knn_color = KNeighborsClassifier(n_neighbors=3)
knn_color.fit(X_color, y_color)

# Predict for the new data point
new_data_point = np.array([[20, 35]])
prediction_color = knn_color.predict(new_data_point)

print(f"The predicted class for Brightness={new_data_point[0][0]} and Saturation={new_data_point[0][1]} is: {prediction_color[0]}")
print("\n")


# 3. Apply k-NN on the iris data set
print("--- Assignment 8.3: k-NN on Iris Dataset ---")
# Load the Iris dataset
iris = load_iris()
X_iris, y_iris = iris.data, iris.target

# Split the data into training and testing sets
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

# k-NN model
knn_iris = KNeighborsClassifier(n_neighbors=3)
knn_iris.fit(X_train_iris, y_train_iris)

# Make predictions on the test set
y_pred_iris = knn_iris.predict(X_test_iris)

# Evaluate the model
accuracy_iris = accuracy_score(y_test_iris, y_pred_iris)
print(f"Accuracy of the k-NN classifier on the Iris dataset is: {accuracy_iris}")
