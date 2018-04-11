#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:13:49 2018

@author: lakshay
"""

import numpy as np
import math

def create_dataset():
    dataset = [[1, 1, 'yes'],
               [1, 1, 'yes'],
               [1, 0, 'no'],
               [0, 1, 'no'],
               [0, 1, 'no']]
    feature_labels = ['No Surfacing', 'Flippers']
    return dataset, feature_labels
               

def calcShannonEntropy(dataset):
    numEntries = len(dataset)
    classCount = {}
    for entry in dataset:
        label = entry[-1]
        classCount[label] = classCount.get(label, 0) + 1
    #print(classCount)
    shannonEntropy = 0.0
    for key in classCount:
        #print(key)
        probability = float(classCount[key]) / numEntries
        shannonEntropy = shannonEntropy + probability * math.log2(probability)
    shannonEntropy = (-1.0) * shannonEntropy
    return shannonEntropy


dataset, feat_labels = create_dataset()
print(calcShannonEntropy(dataset))


def splitDataset(dataset, axis, value):
    newDataset = []
    for featureVector in dataset:
        if featureVector[axis] == value:
            reducedFeatureVector = featureVector[:axis]
            reducedFeatureVector.extend(featureVector[axis+1:])
            newDataset.append(reducedFeatureVector)
    return newDataset


print(dataset)
print(splitDataset(dataset, 0, 0))
print(splitDataset(dataset, 0, 1))
print(splitDataset(dataset, 1, 0))
print(splitDataset(dataset, 1, 1))

print(calcShannonEntropy(splitDataset(dataset, 0, 0)))
print(calcShannonEntropy(splitDataset(dataset, 0, 1)))
print(calcShannonEntropy(splitDataset(dataset, 1, 0)))
print(calcShannonEntropy(splitDataset(dataset, 1, 1)))


def chooseBestFeatureToSplit(dataset):
    numFeatures = len(dataset[0]) - 1
    baseEntropy = calcShannonEntropy(dataset)
    bestInfoGain = 0.0
    bestFeatureToSplitOn = -1
    for i in range(numFeatures):
        featureList = [element[i] for element in dataset]
        uniqueVals = set(featureList)
        newEntropy = 0.0
        for value in uniqueVals:
            subDataset = splitDataset(dataset, i, value)
            prob = len(subDataset) / float(len(dataset))
            newEntropy = newEntropy + prob * calcShannonEntropy(subDataset)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeatureToSplitOn = i
    return bestFeatureToSplitOn

print(chooseBestFeatureToSplit(dataset))














