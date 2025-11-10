from sklearn.datasets import load_digits
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.utils import resample

# Assignment 7: Random Forest Algorithm

# 1. Implement Random Forest algorithm on MNIST data
print("--- Assignment 7.1: Random Forest on MNIST ---")
digits = load_digits()
X_digits, y_digits = digits.data, digits.target
X_train_digits, X_test_digits, y_train_digits, y_test_digits = train_test_split(X_digits, y_digits, test_size=0.2, random_state=42)

rf_digits = RandomForestClassifier(n_estimators=100, random_state=42)
rf_digits.fit(X_train_digits, y_train_digits)
y_pred_digits = rf_digits.predict(X_test_digits)

print(f"Accuracy on MNIST test set: {accuracy_score(y_test_digits, y_pred_digits)}")
print("Confusion Matrix:")
print(confusion_matrix(y_test_digits, y_pred_digits))
print("\n")


# 2. Implement Random Forest algorithm on Mental health data
print("--- Assignment 7.2: Random Forest on Mental Health Data ---")
print("NOTE: The mental health dataset was not provided. The following is a placeholder demonstrating the steps.")
# Example placeholder:
# mental_health_df = pd.read_csv('path_to_your_mental_health_data.csv')
# X_mh = mental_health_df.drop('target_variable', axis=1)
# y_mh = mental_health_df['target_variable']
# X_train_mh, X_test_mh, y_train_mh, y_test_mh = train_test_split(X_mh, y_mh, test_size=0.2, random_state=42)
# rf_mh = RandomForestClassifier(n_estimators=100, random_state=42)
# rf_mh.fit(X_train_mh, y_train_mh)
# y_pred_mh = rf_mh.predict(X_test_mh)
# print(f"Accuracy on Mental Health dataset: {accuracy_score(y_test_mh, y_pred_mh)}")
print("\n")


# 3. Implement Random Forest algorithm on customers' default payments data set.
print("--- Assignment 7.3: Random Forest on Credit Card Default Data ---")

# a) Import the required libraries (done at the top)

# b) Load the data set.
# The dataset is in an Excel file, so we need to provide the sheet name.
# The file needs to be downloaded from the URL and placed in the same directory.
# URL: https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients

try:
    # The file is expected to be named 'default of credit card clients.xls'
    # and be in the same directory as the script.
    # The actual data starts from the second row, so we skip the first row.
    df_credit = pd.read_excel('default of credit card clients.xls', sheet_name='Data', header=1)
    print("Credit card default dataset loaded successfully.")

    # c) Explore the data set – drop the ID column
    df_credit = df_credit.drop('ID', axis=1)

    # d) Analyze missing data
    print("\nChecking for missing data...")
    print(df_credit.isnull().sum())

    # Comment on the values in EDUCATION and MARRIAGE columns
    print("\nUnique values in EDUCATION:", df_credit['EDUCATION'].unique())
    print("Unique values in MARRIAGE:", df_credit['MARRIAGE'].unique())
    # Filtering the rows where the EDUCATION and MARRIAGE columns have non-zero values
    df_credit = df_credit[(df_credit['EDUCATION'] != 0) & (df_credit['MARRIAGE'] != 0)]

    # e) check whether the target variable is balanced using a plot
    plt.figure(figsize=(6, 4))
    df_credit['default payment next month'].value_counts().plot(kind='bar')
    plt.title('Target Variable Distribution')
    plt.xlabel('Default Payment Next Month')
    plt.ylabel('Count')
    plt.show()

    # f) down sample the data
    df_majority = df_credit[df_credit['default payment next month'] == 0]
    df_minority = df_credit[df_credit['default payment next month'] == 1]

    df_majority_downsampled = resample(df_majority, 
                                     replace=False,    # sample without replacement
                                     n_samples=len(df_minority), # to match minority class
                                     random_state=42) # reproducible results

    df_downsampled = pd.concat([df_majority_downsampled, df_minority])

    # g) Hot encode the independent variables
    X = df_downsampled.drop('default payment next month', axis=1)
    y = df_downsampled['default payment next month']
    X_encoded = pd.get_dummies(X, columns=['SEX', 'EDUCATION', 'MARRIAGE'], drop_first=True)

    # h) Split the data set
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # i) Classify accounts and evaluate the model
    rf_credit = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_credit.fit(X_train, y_train)
    y_pred = rf_credit.predict(X_test)

    print(f"\nAccuracy on credit card default test set: {accuracy_score(y_test, y_pred)}")
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))

    # j) Optimize the model with hyperparameter tuning
    param_grid = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_leaf': [1, 2, 4],
        'min_samples_split': [2, 5, 10]
    }
    # Due to the long execution time, we are only showing the setup for GridSearchCV.
    # To run it, uncomment the following lines.
    # grid_search = GridSearchCV(estimator=rf_credit, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2)
    # grid_search.fit(X_train, y_train)
    # print("Best parameters found: ", grid_search.best_params_)
    # best_grid = grid_search.best_estimator_
    # y_pred_best = best_grid.predict(X_test)
    # print(f'Accuracy with best parameters: {accuracy_score(y_test, y_pred_best)}')
    # print("Confusion matrix with best parameters:")
    # print(confusion_matrix(y_test, y_pred_best))

except FileNotFoundError:
    print("\nERROR: 'default of credit card clients.xls' not found.")
    print("Please download the dataset from the URL provided in the assignment and place it in the same directory as this script.")
