#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:08:21 2018

@author: lakshay
"""

import matplotlib.pyplot as plt

"""
plt.scatter(0.2, 0.1)
plt.scatter(1, 1)
plt.annotate("Annotation for the point", xy=(0.2, 0.1), xytext=(0.5, 0.4), arrowprops=dict(arrowstyle="->"))
"""

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xytext=centerPt, textcoords='axes fraction',\
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

def createPlot():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot.ax1 = plt.subplot(1, 1, 1, frameon=False)
    plotNode('a decision node', (0.5, 0.1), (0.5, 0.1), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

def getNumLeaves(myTree):
    if isinstance(myTree, dict) == False: # Which means that is a leaf node.
        return 1
    num_leaves = 0
    for key in myTree:
        num_leaves = num_leaves + getNumLeaves(myTree[key])
    return num_leaves

def plotTree_simple(myTree, centerPt, parentPt):
    if isinstance(myTree, dict) == False:
        plotNode(myTree, centerPt, parentPt, leafNode)
        return
    label = list(myTree.keys())[0]
    plotNode(label, centerPt, parentPt, decisionNode)
    labelValDict = myTree[label]
    i = 0
    for key in labelValDict:
        createPlot.ax1.text(centerPt[0] + 0.1*i, centerPt[1] - 0.1, key)
        plotTree_simple(labelValDict[key], (centerPt[0] + 0.2*i, centerPt[1] - 0.2), centerPt)
        i = i + 1
        
def createTreePlot_simple(myTree):
    fig = plt.figure()
    fig.clf()
    createPlot.ax1 = plt.subplot(1, 1, 1, frameon=False)
    plotTree_simple(myTree, (0.5, 0.5), (0.5, 0.5))
    plt.tick_params()
    plt.show()
    
