#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  9 13:03:33 2018

@author: lakshay
"""
import numpy as np
import operator
import os

def img2vector(filename):
    returnVec = np.zeros((1024))
    f = open(filename)
    for i in range(32):
        line = f.readline()
        for j in range(32):
            returnVec[32*i + j] = int(line[j])
    return returnVec

def load_data():

    train_dir = "./data/digits/trainingDigits/"
    test_dir = "./data/digits/testDigits/"
    
    train_files = os.listdir(train_dir)
    test_files = os.listdir(test_dir)
    
    X_train = []
    y_train = []
    for file_path in train_files:
        X_train.append(img2vector(train_dir + file_path))
        y_train.append(int(file_path.split('_')[0]))

    X_test = []
    y_test = []
    for file_path in test_files:
        X_test.append(img2vector(test_dir + file_path))
        y_test.append(int(file_path.split('_')[0]))

    X_train = np.array(X_train)
    X_test = np.array(X_test)
    return X_train, y_train, X_test, y_test

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

def handwritingClassTest():
    X_train, y_train, X_test, y_test = load_data()
    errorCount = 0
    for i in range(X_test.shape[0]):
        prediction = classify0(X_test[i], X_train, y_train, 3)
        print("Predicted Label: %d    Ground-Truth Label: %d" % (prediction, y_test[i]))
        if prediction != y_test[i]:
            errorCount = errorCount + 1
    errorRate = errorCount / X_test.shape[0]
    print("Error Rate: ", errorRate)
    print("Accuracy: ", 100 - errorRate*100)
        