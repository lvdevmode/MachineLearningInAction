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
    plt.annotate(y[i], xy=X[i], xytext=(X[i][0] + 0.02, X[i][1]))
plt.show()


