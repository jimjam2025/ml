import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris, load_diabetes
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import fetch_openml

# Assignment 4: Implementing Neural Networks

# a) Design and implement a neural network that acts as a AND classifier for binary input
print("--- Assignment 4a: AND Gate Classifier ---")
X_and = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_and = np.array([0, 0, 0, 1])

model_and = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_dim=2)
])
model_and.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_and.fit(X_and, y_and, epochs=1000, verbose=0)
print("AND Gate Predictions:")
print(model_and.predict(X_and).round())
print("\n")

# b) Design and implement a neural network that acts as a OR classifier for binary input
print("--- Assignment 4b: OR Gate Classifier ---")
X_or = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_or = np.array([0, 1, 1, 1])

model_or = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_dim=2)
])
model_or.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_or.fit(X_or, y_or, epochs=1000, verbose=0)
print("OR Gate Predictions:")
print(model_or.predict(X_or).round())
print("\n")

# c) Design and implement a neural network that acts as a NAND classifier for binary input
print("--- Assignment 4c: NAND Gate Classifier ---")
X_nand = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_nand = np.array([1, 1, 1, 0])

model_nand = tf.keras.models.Sequential([
    tf.keras.layers.Dense(1, activation='sigmoid', input_dim=2)
])
model_nand.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_nand.fit(X_nand, y_nand, epochs=1000, verbose=0)
print("NAND Gate Predictions:")
print(model_nand.predict(X_nand).round())
print("\n")

# d) Design and implement a neural network that acts as a XOR classifier for binary input
print("--- Assignment 4d: XOR Gate Classifier ---")
X_xor = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
y_xor = np.array([0, 1, 1, 0])

model_xor = tf.keras.models.Sequential([
    tf.keras.layers.Dense(8, activation='relu', input_dim=2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])
model_xor.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_xor.fit(X_xor, y_xor, epochs=1000, verbose=0)
print("XOR Gate Predictions:")
print(model_xor.predict(X_xor).round())
print("Comment: A single-layer neural network (perceptron) cannot classify the XOR data because it is not linearly separable. A multi-layer network with a hidden layer is required to create a non-linear decision boundary.")
print("\n")

# e) Design and implement a sequential, dense neural network that acts as a classifier for the Iris dataset
print("--- Assignment 4e: Iris Dataset Classifier ---")
iris = load_iris()
X_iris, y_iris = iris.data, iris.target
X_train_iris, X_test_iris, y_train_iris, y_test_iris = train_test_split(X_iris, y_iris, test_size=0.2, random_state=42)

scaler_iris = StandardScaler()
X_train_iris = scaler_iris.fit_transform(X_train_iris)
X_test_iris = scaler_iris.transform(X_test_iris)

model_iris = tf.keras.models.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(X_train_iris.shape[1],)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# Tuning hyperparameters
learning_rate = 0.01
epochs = 100

optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
model_iris.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model_iris.fit(X_train_iris, y_train_iris, epochs=epochs, verbose=0)
loss_iris, accuracy_iris = model_iris.evaluate(X_test_iris, y_test_iris, verbose=0)
print(f'Iris dataset accuracy: {accuracy_iris}')
print("\n")

# f) Design and implement a sequential, dense neural network that acts as a classifier for the diabetes dataset
print("--- Assignment 4f: Diabetes Dataset Classifier ---")
diabetes = load_diabetes()
X_diabetes, y_diabetes = diabetes.data, diabetes.target
# For classification, we can create a binary target variable (e.g., above or below median)
y_diabetes_binary = (y_diabetes > np.median(y_diabetes)).astype(int)

X_train_diabetes, X_test_diabetes, y_train_diabetes, y_test_diabetes = train_test_split(X_diabetes, y_diabetes_binary, test_size=0.2, random_state=42)

scaler_diabetes = StandardScaler()
X_train_diabetes = scaler_diabetes.fit_transform(X_train_diabetes)
X_test_diabetes = scaler_diabetes.transform(X_test_diabetes)

model_diabetes = tf.keras.models.Sequential([
    tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train_diabetes.shape[1],)),
    tf.keras.layers.Dense(8, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_diabetes.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_diabetes.fit(X_train_diabetes, y_train_diabetes, epochs=150, verbose=0)
loss_diabetes, accuracy_diabetes = model_diabetes.evaluate(X_test_diabetes, y_test_diabetes, verbose=0)
print(f'Diabetes dataset accuracy: {accuracy_diabetes}')
print("\n")

# g) Design and implement a sequential, dense neural network that acts as a classifier for the heart dataset
print("--- Assignment 4g: Heart Disease Dataset Classifier ---")
# Using the Heart Disease dataset from OpenML
heart = fetch_openml('heart', version=1, as_frame=True)
X_heart = heart.data
y_heart = (heart.target == '2').astype(int) # Target: 1 for disease, 0 for no disease

X_train_heart, X_test_heart, y_train_heart, y_test_heart = train_test_split(X_heart, y_heart, test_size=0.2, random_state=42)

scaler_heart = StandardScaler()
X_train_heart = scaler_heart.fit_transform(X_train_heart)
X_test_heart = scaler_heart.transform(X_test_heart)

model_heart = tf.keras.models.Sequential([
    tf.keras.layers.Dense(20, activation='relu', input_shape=(X_train_heart.shape[1],)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model_heart.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
model_heart.fit(X_train_heart, y_train_heart, epochs=200, verbose=0)
loss_heart, accuracy_heart = model_heart.evaluate(X_test_heart, y_test_heart, verbose=0)
print(f'Heart disease dataset accuracy: {accuracy_heart}')
