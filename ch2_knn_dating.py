#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:52:17 2018

@author: lakshay
"""

import numpy as np
import operator

# Parsing the input text file

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
    
    X = np.array(X)
    return X, y


# Analyzing the input / Visualization
 
X, y = file2matrix("./data/datingTestSet_withNumericLabels.txt")
## X = [Number of Frequent Flyer Miles Earned Per Year, Percentage of Time Spent Playing Video Games,
## Liters of Ice-Cream Consumed Per Week], y = [Degree of Likenesss]

import matplotlib.pyplot as plt

fig = plt.figure()

ax1 = fig.add_subplot(2, 1, 1)
ax1.scatter(X[:, 1], X[:, 2])
ax1.set_xlabel("Percentage of Time Spent Playing Video Games")
ax1.set_ylabel("Liters of Ice-Cream Consumed Per Week")

ax2 = fig.add_subplot(2, 1, 2)
ax2.scatter(X[:, 1], X[:, 2], 15.0*np.array(y), 15.0*np.array(y))
ax2.set_xlabel("Percentage of Time Spent Playing Video Games")
ax2.set_ylabel("Liters of Ice-Cream Consumed Per Week")

plt.show()