#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr 17 11:23:37 2018

@author: lakshay
"""

import os
import re
import numpy as np

def textParse(bigString):
    string = str(bigString)
    listOfTokens = re.split(r'\W*', string)
    return [tok.lower() for tok in listOfTokens if len(tok) > 2]

def load_data():
    dataDir = './data/email/'
    uniqueLabels = os.listdir(dataDir)
    datasetFiles = []
    labels = []
    for label in uniqueLabels:
        subDir = dataDir + label
        files = [subDir + '/' + file for file in os.listdir(subDir)]
        datasetFiles = datasetFiles + files
        labels = labels + [label] * len(files)
    
    from sklearn.preprocessing import LabelEncoder
    enc = LabelEncoder()
    numericLabels = enc.fit_transform(labels)
    
    dataset = []
    for file in datasetFiles:
        f = open(file, 'r', encoding='latin1')
        dataset.append(textParse(f.read()))
        f.close()
    
    return dataset, numericLabels, labels

def createVocabList(dataset):
    vocab = set() # As we need all the unique words in our dataset, we use a set.
    for document in dataset:
        vocab =  vocab | set(document) # Union of two sets.
    return list(vocab)

def bagOfWordsToVec(vocab, inputSet): # inputSet is the set of unique words in the input document.
    vec = [0] * len(vocab)
    for word in inputSet:
        if word in vocab:
            vec[vocab.index(word)] = vec[vocab.index(word)] + 1
        else:
            print("The word '%s' is not in the vocabulary." % (word))
    return vec

def trainNaiveBayes(dataset, labels):
    numOfDocs = len(dataset)
    vocab = createVocabList(dataset)
    uniqueLabels = list(set(labels))
    countWordsLabelwise = np.ones((len(uniqueLabels), len(vocab))) #
    labelCount = np.zeros((len(uniqueLabels), 1))
    for i in range(numOfDocs):
        document = dataset[i]
        label = labels[i]
        wordVec = bagOfWordsToVec(vocab, document)
        countWordsLabelwise[label] = countWordsLabelwise[label] + np.array(wordVec)
        labelCount[label] = labelCount[label] + 1
    totalWordsLabelwise = np.sum(countWordsLabelwise, 1).reshape((-1, 1)) + np.array([[2], [2]]) #
    probLabel = labelCount / numOfDocs
    probWordLabel = np.log(countWordsLabelwise / totalWordsLabelwise) #
    return probWordLabel, probLabel    

def classifyNaiveBayes(vec2Classify, probWordLabel, probLabel):
    probClass = np.sum(np.array(vec2Classify) * probWordLabel, axis=1).reshape(-1, 1) \
                + np.log(probLabel) # log(a) + log(b) = log(a*b)
    return np.argmax(probClass)

def testingNaiveBayes():
    dataset, labels, labels_ = load_data()
    vocab = createVocabList(dataset)
    probWordLabel, probLabel = trainNaiveBayes(dataset, labels)