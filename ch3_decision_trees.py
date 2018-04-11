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
        probability = float(classCount[key]) / numEntries
        shannonEntropy = shannonEntropy + probability * math.log2(probability)
    shannonEntropy = (-1.0) * shannonEntropy
    return shannonEntropy

#dataset, feat_labels = create_dataset()
#print(calcShannonEntropy(dataset))