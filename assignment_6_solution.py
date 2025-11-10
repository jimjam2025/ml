import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.feature_extraction.text import CountVectorizer

# Assignment 6: Implementation of the Gaussian Naïve Bayes Classifier

# 1. Design and implement the naïve Bayes classifier using the Adult dataset
print("--- Assignment 6.1: Gaussian Naïve Bayes on Adult Dataset ---")

# Load the dataset
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data'
df = pd.read_csv(url, header=None, sep=r',\s*', engine='python')
df.columns = ['age', 'workclass', 'fnlwgt', 'education', 'education-num', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'capital-gain', 'capital-loss', 'hours-per-week', 'native-country', 'income']

# a) Check for missing values and frequency counts
print("\nInitial info:")
df.info()

# b) Replace ? with NaN
df.replace('?', np.nan, inplace=True)

# c) Check labels and frequency distribution
print("\nFrequency distribution of workclass:")
print(df['workclass'].value_counts())

# d) Impute missing categorical variables
categorical_cols_with_nan = ['workclass', 'occupation', 'native-country']
for col in categorical_cols_with_nan:
    df[col].fillna(df[col].mode()[0], inplace=True)

# e) Explore numerical variables
print("\nNumerical variables description:")
print(df.describe())

# f) Declare feature and target variables
X = df.drop(['income'], axis=1)
y = df['income']

# g) Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# h) One-hot encoding
categorical_cols = ['workclass', 'education', 'marital-status', 'occupation', 'relationship', 'race', 'sex', 'native-country']
X_train = pd.get_dummies(X_train, columns=categorical_cols, drop_first=True)
X_test = pd.get_dummies(X_test, columns=categorical_cols, drop_first=True)

# Align columns - crucial for consistent feature sets
train_cols = X_train.columns
test_cols = X_test.columns
missing_in_test = set(train_cols) - set(test_cols)
for c in missing_in_test:
    X_test[c] = 0
missing_in_train = set(test_cols) - set(train_cols)
for c in missing_in_train:
    X_train[c] = 0
X_test = X_test[train_cols]

# i) Feature Scaling
scaler = RobustScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# j) Fit the GaussianNB model
gnb = GaussianNB()
gnb.fit(X_train_scaled, y_train)

# k) Predict on the test data
y_pred = gnb.predict(X_test_scaled)

# l) Print model accuracy
print(f"\nModel Accuracy: {accuracy_score(y_test, y_pred)}")

# m) Check for overfitting and underfitting
# A simple check is to compare training and testing accuracy
print(f"Training Accuracy: {gnb.score(X_train_scaled, y_train)}")
print(f"Testing Accuracy: {gnb.score(X_test_scaled, y_test)}")

# n) Compare model accuracy with null accuracy
ull_accuracy = y_test.value_counts().max() / len(y_test)
print(f"Null Accuracy: {null_accuracy}")

# o) Print the confusion matrix
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))

# p) Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



# 2. Design and implement a Multinomial Naive Bayes Classifier for text classification
print("\n--- Assignment 6.2: Multinomial Naïve Bayes for Text Classification ---")

# Sample dataset
text_data = {
    'text': [
        'The new iPhone has a great camera', 
        'Manchester United wins the league', 
        'The election results are in', 
        'The new Marvel movie is a blockbuster', 
        'The latest GPU is so fast', 
        'Real Madrid signs a new player', 
        'The parliament passed a new bill', 
        'The movie received great reviews'
    ],
    'category': ['Technology', 'Sports', 'Politics', 'Entertainment', 'Technology', 'Sports', 'Politics', 'Entertainment']
}
df_text = pd.DataFrame(text_data)

# Prepare the data
X_text = df_text['text']
y_text = df_text['category']

# Vectorize the text data
vectorizer = CountVectorizer()
X_text_vectorized = vectorizer.fit_transform(X_text)

# Split the data
X_train_text, X_test_text, y_train_text, y_test_text = train_test_split(X_text_vectorized, y_text, test_size=0.25, random_state=42)

# Fit the MultinomialNB model
mnb = MultinomialNB()
mnb.fit(X_train_text, y_train_text)

# Predict on the test data
y_pred_text = mnb.predict(X_test_text)

# Evaluate the model
print(f"\nText Classification Accuracy: {accuracy_score(y_test_text, y_pred_text)}")
print("\nClassification Report:")
print(classification_report(y_test_text, y_pred_text))

# Example of predicting a new sentence
new_sentence = ['The president gave a speech']
new_sentence_vectorized = vectorizer.transform(new_sentence)
predicted_category = mnb.predict(new_sentence_vectorized)
print(f"\nThe sentence '{new_sentence[0]}' is predicted to be in the category: {predicted_category[0]}")
