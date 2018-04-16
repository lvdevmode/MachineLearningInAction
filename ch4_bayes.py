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

#dataset, labels = loadDataset()

def createVocabList(dataset):
    vocab = set() # As we need all the unique words in our dataset, we use a set.
    for document in dataset:
        vocab =  vocab | set(document) # Union of two sets.
    return list(vocab)

#vocab = createVocabList(dataset)

def setOfWordsToVec(vocab, inputSet): # inputSet is the set of unique words in the input document.
    vec = [0] * len(vocab)
    for word in inputSet:
        if word in vocab:
            vec[vocab.index(word)] = 1
        else:
            print("The word '%s' is not in the vocabulary." % (word))
    return vec

#wordVec = setOfWordsToVec(vocab, set(['you', 'are', 'so', 'stupid']))

def trainNaiveBayes_intuitive(dataset, labels):
    numOfDocs = len(dataset)
    vocab = createVocabList(dataset)
    uniqueLabels = list(set(labels))
    countWordsLabelwise = np.zeros((len(uniqueLabels), len(vocab)))
    labelCount = np.zeros((len(uniqueLabels), 1))
    for i in range(numOfDocs):
        document = dataset[i]
        label = labels[i]
        wordVec = setOfWordsToVec(vocab, document)
        countWordsLabelwise[label] = countWordsLabelwise[label] + np.array(wordVec)
        labelCount[label] = labelCount[label] + 1
    totalWordsLabelwise = np.sum(countWordsLabelwise, 1).reshape((-1, 1))
    probLabel = labelCount / numOfDocs
    probWordLabel = countWordsLabelwise / totalWordsLabelwise
    return probWordLabel, probLabel

"""
P(f1, f2, f3 | c) = P(f1 | c) * P(f3 | c) * P(f3 | c) => When f1, f2, and f3 are independent. 
This is called conditional independence, and is one of the assumptions of naive bayes.
"""

#probWordLabel, probLabel = trainNaiveBayes_intuitive(dataset, labels)

"""
Now, when testing, if any one of words present in the test case has P(word | ci) = 0,
the whole thing will become zero because we multiply a lot of probabilities to get
the P(ci | W), where W is a word vector w0, w1, ....
So, in order to lessen the impact of this, we initialize the counts from 1, rather than 0.
And we add 2 to the denominator i.e totalWordsLabelWise.
Also, there is a problem of underflow. Too many multiplications with small numbers, and python
might approximate them to 0. So, we take the log of the probabilities (ln).
"""

def trainNaiveBayes(dataset, labels):
    numOfDocs = len(dataset)
    vocab = createVocabList(dataset)
    uniqueLabels = list(set(labels))
    countWordsLabelwise = np.ones((len(uniqueLabels), len(vocab))) #
    labelCount = np.zeros((len(uniqueLabels), 1))
    for i in range(numOfDocs):
        document = dataset[i]
        label = labels[i]
        wordVec = setOfWordsToVec(vocab, document)
        countWordsLabelwise[label] = countWordsLabelwise[label] + np.array(wordVec)
        labelCount[label] = labelCount[label] + 1
    totalWordsLabelwise = np.sum(countWordsLabelwise, 1).reshape((-1, 1)) + np.array([[2], [2]]) #
    probLabel = labelCount / numOfDocs
    probWordLabel = np.log(countWordsLabelwise / totalWordsLabelwise) #
    return probWordLabel, probLabel    

def classifyNaiveBayes(vec2Classify, probWordLabel, probLabel):
    probClass = np.sum(np.array(vec2Classify) * probWordLabel, axis=1).reshape(-1, 1) + np.log(probLabel)
    return np.argmax(probClass)