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
            prob = len(subDataset) / float(len(dataset)) # See following explanation.
            newEntropy = newEntropy + prob * calcShannonEntropy(subDataset)
        infoGain = baseEntropy - newEntropy
        if infoGain > bestInfoGain:
            bestInfoGain = infoGain
            bestFeatureToSplitOn = i
    return bestFeatureToSplitOn

"""
So, in this loop, what are we doing is that for each unique value the feature i can take
we create the split and then calculate entropy for that split. But now, in order to get 
the collective entropy i.e. total entropy for that feature as a whole, we cannot directly
sum up all the entropies for the different splits of that feature. Rather, we multiply
entropy calculated for the particular value of that feature vector, and then multiply
it with the probability of that particular value occuring in that feature.
"""

print(chooseBestFeatureToSplit(dataset))

"""
We observe that the above command returns 0 as the best feature to split on. It means 
feature 0 gives best information gain or reduction in entropy. Reduction in entropy
would mean decrease in the randomness of the data, while at the same time, gain in
information means that now the data is more organized. 
This can be clearly seen by comparing the splits in case of feature 0 and feature 1.
When the split is done using feature 0, we see that we get 2 subsets. One comprising of
the labels [yes, yes, no], and the other one containing [no, no]. In case of feature 1,
we get subsets containing labels [no] and [yes , yes, no, no]. This split, using
feature 0, is clearly less organized as compared to feature 0, as our target is to divide
the dataset such that each subset contains only one kind of label.
"""













