import numpy as np

# Assignment 5: Find-S and CEA Implementation

# Sample dataset (e.g., Enjoy Sport)
# Attributes: Sky, AirTemp, Humidity, Wind, Water, Forecast, EnjoySport
data = np.array([
    ['Sunny', 'Warm', 'Normal', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Warm', 'Same', 'Yes'],
    ['Rainy', 'Cold', 'High', 'Strong', 'Warm', 'Change', 'No'],
    ['Sunny', 'Warm', 'High', 'Strong', 'Cool', 'Change', 'Yes']
])

attributes = ['Sky', 'AirTemp', 'Humidity', 'Wind', 'Water', 'Forecast']
X = data[:, :-1]
y = data[:, -1]

# a) Implement the Find-S algorithm
print("--- Assignment 5a: Find-S Algorithm ---")

def find_s(X, y):
    # Initialize with the most specific hypothesis
    hypothesis = ['0'] * len(attributes)
    
    # Get the indices of positive examples
    positive_examples_indices = np.where(y == 'Yes')[0]
    
    # The first positive example is the initial hypothesis
    for i in range(len(attributes)):
        hypothesis[i] = X[positive_examples_indices[0]][i]
        
    # Generalize the hypothesis with other positive examples
    for i in positive_examples_indices[1:]:
        for j in range(len(attributes)):
            if X[i][j] != hypothesis[j]:
                hypothesis[j] = '?'
    return hypothesis

find_s_hypothesis = find_s(X, y)
print("Final Hypothesis (Find-S):", find_s_hypothesis)
print("\n")


# b) Implement the Candidate Elimination (List-Then-Eliminate) algorithm
print("--- Assignment 5b: Candidate-Elimination Algorithm ---")

def candidate_elimination(X, y):
    num_attributes = len(attributes)
    
    # Initialize G to the most general hypothesis
    G = [[ '?' for _ in range(num_attributes)]]
    
    # Initialize S to the most specific hypothesis from the first positive example
    positive_examples_indices = np.where(y == 'Yes')[0]
    S = [X[positive_examples_indices[0]].tolist()]
    
    for i, example in enumerate(X):
        if y[i] == 'Yes': # Positive example
            # Remove from G any hypothesis inconsistent with the example
            G = [g for g in G if all(g[j] == '?' or g[j] == example[j] for j in range(num_attributes))]
            
            # For each hypothesis in S that is not consistent with the example, remove it
            # and add its minimal generalizations that are consistent and more specific than some hypothesis in G
            for s in S[:]: # Iterate over a copy
                if not all(s[j] == example[j] for j in range(num_attributes)):
                    S.remove(s)
                    for j in range(num_attributes):
                        if s[j] != example[j]:
                            new_h = s[:]
                            new_h[j] = '?'
                            # Check if new_h is more specific than some hypothesis in G
                            if any(all(G[k][l] == '?' or G[k][l] == new_h[l] for l in range(num_attributes)) for k in range(len(G))):
                                if new_h not in S:
                                    S.append(new_h)
        else: # Negative example
            # Remove from S any hypothesis inconsistent with the example
            S = [s for s in S if not all(s[j] == '?' or s[j] == example[j] for j in range(num_attributes))]
            
            # For each hypothesis in G that is not consistent with the example, remove it
            # and add its minimal specializations that are consistent and more specific than some hypothesis in S
            for g in G[:]: # Iterate over a copy
                if all(g[j] == '?' or g[j] == example[j] for j in range(num_attributes)):
                    G.remove(g)
                    for j in range(num_attributes):
                        if g[j] == '?':
                            # Get unique values for the attribute from the training data
                            values = np.unique(X[:, j])
                            for val in values:
                                if val != example[j]:
                                    new_h = g[:]
                                    new_h[j] = val
                                    # Check if new_h is more general than some hypothesis in S
                                    if any(all(new_h[l] == '?' or new_h[l] == S[k][l] for l in range(num_attributes)) for k in range(len(S))):
                                        if new_h not in G:
                                            G.append(new_h)
    return S, G

S, G = candidate_elimination(X, y)
print("Final S (Most Specific Hypotheses):", S)
print("Final G (Most General Hypotheses):", G)
