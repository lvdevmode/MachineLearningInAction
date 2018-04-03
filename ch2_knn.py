#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 00:18:11 2018

@author: lakshay
"""

import numpy as np
import operator

def create_dataset():
    group = np.array([[1.0, 1.1], [1.0, 1.0], [0, 0], [0, 0.1]])
    labels = ['A', 'A', 'B', 'B']
    labels_ = [0, 0, 1 , 1] # for visualization
    return group, labels, labels_



# Visualization
import matplotlib.pyplot as plt
X, y, y_ = create_dataset()

plt.scatter(X[:, 0], X[:, 1], c=y_) # c requires a numerical value.
for i in range(X.shape[0]):
    plt.annotate(y[i], xy=X[i], xytext=(X[i][0] + 0.02, X[i][1])) # Placing the label 0.02 units to the right.
plt.show()




# Crude kNN implementation - Intuition based.
def classify0_basic(inX, dataset, labels, k):
    n = dataset.shape[0]
    labels = np.array(labels)
    distances = np.zeros(n)
    
    for i in range(n):
        p = dataset[i]
        distances[i] = np.sqrt((inX[0] - p[0])*(inX[0] - p[0]) + (inX[1] - p[1])*(inX[1] - p[1]))
    
    sorted_permutation = distances.argsort() # Returns the indices that would sort this array.
    distances = distances[sorted_permutation]
    labels = labels[sorted_permutation]
    print("Sorted Distances: ", distances)
    print("Respective Labels: ", labels)
    
    votes = {}
    max_count = 0
    prediction = ''
    for i in range(k):
        if labels[i] not in votes:
            votes[labels[i]] = 1
        else:
            votes[labels[i]] = votes[labels[i]] + 1
        if votes[labels[i]] > max_count:
            max_count = votes[labels[i]]
            prediction = labels[i]
    print("Votes: ", votes)
    return prediction

X, y, y_ = create_dataset()
inX = [0.1, 0.9]
plt.scatter(inX[0], inX[1])
k = 3
prediction = classify0_basic(inX, X, y, k)
print("Predicted Label for ", inX, " is ", prediction)




# Better implementation
def classify0(inx, dataset, labels, k):
    n = dataset.shape[0]
    diffMat = np.tile(inx, (n, 1)) - dataset # Look into tile. Quite useful.
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    print("Sorted Distances: ", distances[sortedDistIndices])
    print("Respective Labels: ", np.array(labels)[sortedDistIndices])
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) # Sort using iterators. Important.
    print("Votes: ", sortedClassCount)
    return sortedClassCount[0][0]

X, y, y_ = create_dataset()
inX = [0.1, 0.9]
plt.scatter(inX[0], inX[1])
k = 3
prediction = classify0(inX, X, y, k)
print("Predicted Label for ", inX, " is ", prediction)