#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:52:17 2018

@author: lakshay
"""

import numpy as np
import operator

######## Parsing the input text file

## filename = "./data/datingTestSet.txt"
## filename = "./data/datingTestSet_withNumericLabels.txt"

def file2matrix(filename):
    f = open(filename, 'r')

    lines = f.readlines()
    X = []
    y = []
    for line in lines:
        line = line.strip()
        elements = line.split('\t')
        X.append(elements[0:3])
        y.append(int(elements[-1]))
    
    X = np.array(X).astype(np.float32)
    return X, y

######## Analyzing the input / Visualization

#X, y = file2matrix("./data/datingTestSet_withNumericLabels.txt")
## X = [Number of Frequent Flyer Miles Earned Per Year, Percentage of Time Spent Playing Video Games,
## Liters of Ice-Cream Consumed Per Week], y = [Degree of Likenesss]

import matplotlib.pyplot as plt

fig = plt.figure()

# Simple plot
ax1 = fig.add_subplot(2, 2, 1)
ax1.scatter(X[:, 1], X[:, 2])
ax1.set_xlabel("Percentage of Time Spent Playing Video Games")
ax1.set_ylabel("Liters of Ice-Cream Consumed Per Week")

# Different color markers of different sizes for different labels.
ax2 = fig.add_subplot(2, 2, 2)
ax2.scatter(X[:, 1], X[:, 2], s=15.0*np.array(y), c=y)
ax2.set_xlabel("Percentage of Time Spent Playing Video Games")
ax2.set_ylabel("Liters of Ice-Cream Consumed Per Week")

# Simply providing labels like this doesn't really work. See for yourself by un-commenting #58.
ax3 = fig.add_subplot(2, 2, 3)
ax3.scatter(X[:, 1], X[:, 0], label=y, c=y)
#ax3.legend()
ax3.set_xlabel("Percentage of Time Spent Playing Video Games")
ax3.set_ylabel("Number of Frequent Flyer Miles Earned Per Year")

# In python, if you need a proper legend, class-wise, then you need to
# plot the points class-wise. And then the options open up. 
ax4 = fig.add_subplot(2, 2, 4)
possible_labels = np.unique(np.array(y))
#markers = ['^', 's', 'o']
X_class = []
for label in possible_labels:
    for i in range(len(y)):
        if y[i] == label :
            X_class.append(X[i])
    X_class = np.array(X_class)
    #ax4.scatter(X_class[:, 2], X_class[:, 0], marker=markers[label - 1], label=label)
    ax4.scatter(X_class[:, 2], X_class[:, 0], label=label)
    X_class = []
ax4.legend()

plt.show()

######## Prepare / Normalizing the feature values

# To normalize the values of a particular feature to a range of 0 to 1, the following
# is used => newVal = (oldVal - minVal) / (maxVal - minVal)

def autonorm_intuitive(X):
    minVal = X.min(0)
    maxVal = X.max(0)
    ranges = maxVal - minVal
    X = (X - minVal) / ranges
    return X, minVal, ranges # minVal, ranges will be used to normalize the test set.

def autonorm(X):
    minVal = X.min(0)
    maxVal = X.max(0)
    ranges = maxVal - minVal
    n = X.shape[0]
    X = X - np.tile(minVal, (n, 1))
    X = X / np.tile(ranges, (n, 1))
    return X, minVal, ranges

#X_normalized = autonorm_intuitive(X)
#X_normalized = autonorm(X)

def classify0(inx, dataset, labels, k):
    n = dataset.shape[0]
    diffMat = np.tile(inx, (n, 1)) - dataset # Look into tile. Quite useful.
    sqDiffMat = diffMat**2
    sqDistances = sqDiffMat.sum(axis=1)
    distances = sqDistances**0.5
    sortedDistIndices = distances.argsort()
    classCount = {}
    for i in range(k):
        voteLabel = labels[sortedDistIndices[i]]
        classCount[voteLabel] = classCount.get(voteLabel, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True) # Sort using iterators. Important.
    return sortedClassCount[0][0]

def datingClassTest():
    X, y = file2matrix("./data/datingTestSet_withNumericLabels.txt")
    normX, minVal, ranges = autonorm_intuitive(X)
    trainTestRatio = 0.1
    n = normX.shape[0]
    numTestVecs = int(trainTestRatio * n)
    errorCount = 0
    for i in range(numTestVecs):
        prediction = classify0(normX[i], normX[numTestVecs:], y[numTestVecs:], 4)
        #print("Predicted Label: %d   Ground Truth Label: %d" % (prediction, y[i]))
        if prediction != y[i]:
            errorCount = errorCount + 1
    errorRate = errorCount / numTestVecs
    print("Error Rate: ", errorRate)
    print("Accuracy: %f" % (100.0 - errorRate * 100))

#datingClassTest()