#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 13 18:22:27 2018

@author: lakshay
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 10 11:13:49 2018

@author: lakshay
"""

import math
import operator
import ch3_treePlotter

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


def splitDataset(dataset, axis, value):
    newDataset = []
    for featureVector in dataset:
        if featureVector[axis] == value:
            newDataset.append(featureVector[:axis] + featureVector[axis+1:])
    return newDataset

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

def majorityCount(classList): # Passing a class list here as this func will be only be called
    classCount =  {}          # when there are no more attributes left to split the dataset on.
    for label in classList:   # (or in special cases, when the number of splits have exceeded a limit)
        classCount[label] = classCount.get(label, 0) + 1
    sortedClassCount = sorted(classCount.items(), key=operator.itemgetter(1), reverse=True)
    return sortedClassCount[0][0]

def createTree(dataset, feat_labels):
    classList = [element[-1] for element in dataset]
    if classList.count(classList[0]) == len(classList):
        return classList[0] # Returns a leaf node, containg only the label.
    if len(dataset[0]) == 1:
        return majorityCount(classList) # Returns a leaf node, containg only the label.
    bestFeatureToSplit = chooseBestFeatureToSplit(dataset)
    bestFeatureLabel = feat_labels[bestFeatureToSplit]
    del(feat_labels[bestFeatureToSplit]) # This is required as the selected feature will be removed from
    myTree = {bestFeatureLabel:{}}       # the dataset, so its label should also be removed.
    featureValues = [element[bestFeatureToSplit] for element in dataset]
    uniqueVals = set(featureValues)
    for val in uniqueVals:
        copy_of_feat_labels = feat_labels[:] # Lists are passed by reference in python, so need a copy.
        split = splitDataset(dataset, bestFeatureToSplit, val)
        myTree[bestFeatureLabel][val] = createTree(split, copy_of_feat_labels)
    return myTree

def load_data():
    f = open('./data/lenses.txt')
    dataset = [line.strip().split('\t') for line in f.readlines()]
    featureLabels = ['age', 'prescript', 'astigmatic', 'tearRate']
    return dataset, featureLabels

def lenses_decision_tree():
    dataset, featureLabels = load_data()
    myTree = createTree(dataset, featureLabels)
    ch3_treePlotter.createTreePlot_simple(myTree)
    ch3_treePlotter.createPlot(myTree)
    

