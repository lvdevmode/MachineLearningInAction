#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Apr 15 22:48:04 2018

@author: lakshay
"""

import numpy as np

def loadDataset():
    postingList = [['my', 'dog', 'has', 'flea', 'problems', 'help', 'please'], 
                   ['maybe', 'not', 'take', 'him', 'to', 'dog', 'park', 'stupid'], 
                   ['my', 'dalmation', 'is', 'so', 'cute', 'I', 'love', 'him'], 
                   ['stop', 'posting', 'stupid', 'worthless', 'garbage'], 
                   ['mr', 'licks', 'ate', 'my', 'steak', 'how', 'to', 'stop', 'him'], 
                   ['quit', 'buying', 'worthless', 'dog', 'food', 'stupid']]
    labels = [0, 1, 0, 1, 0, 1] # 1 is abusive, 0 is normal
    return postingList, labels

dataset, labels = loadDataset()

def createVocabList(dataset):
    vocab = set() # As we need all the unique words in our dataset, we use a set.
    for document in dataset:
        vocab =  vocab | set(document) # Union of two sets.
    return list(vocab)

vocab = createVocabList(dataset)

def setOfWordsToVec(vocab, inputSet): # inputSet is the set of unique words in the input document.
    vec = [0] * len(vocab)
    for word in inputSet:
        if word in vocab:
            vec[vocab.index(word)] = 1
        else:
            print("The word '%s' is not in the vocabulary." % (word))
    return vec

wordVec = setOfWordsToVec(vocab, set(['you', 'are', 'so', 'stupid']))