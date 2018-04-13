#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Apr 12 16:08:21 2018

@author: lakshay
"""

def retrieveTree(i):
    listOfTrees = [{'No Surfacing' : {0: 'no', 1: {'Flippers': {0: 'no', 1: 'yes'}}}}, 
                   {'No Surfacing' : {0: 'no', 1: {'Flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}}},
                   {'No Surfacing' : {0: 'no', 1: {'Flippers': {0: {'head': {0: 'no', 1: 'yes'}}, 1: 'no'}}, 3: 'maybe'}}]
    return listOfTrees[i]

#########################################################

import matplotlib.pyplot as plt

"""
plt.scatter(0.2, 0.1)
plt.scatter(1, 1)
plt.annotate("Annotation for the point", xy=(0.2, 0.1), xytext=(0.5, 0.4), arrowprops=dict(arrowstyle="->"))
"""

decisionNode = dict(boxstyle="sawtooth", fc="0.8")
leafNode = dict(boxstyle="round", fc="0.8")
arrow_args = dict(arrowstyle="<-")

def plotNode_basic(nodeTxt, centerPt, parentPt, nodeType):
    createPlot_basic.ax1.annotate(nodeTxt, xy=parentPt, xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

def createPlot_basic():
    fig = plt.figure(1, facecolor='white')
    fig.clf()
    createPlot_basic.ax1 = plt.subplot(1, 1, 1, frameon=False)
    plotNode('a decision node', (0.5, 0.1), (0.5, 0.1), decisionNode)
    plotNode('a leaf node', (0.8, 0.1), (0.3, 0.8), leafNode)
    plt.show()

################################################

def getNumLeaves_basic(myTree):
    if isinstance(myTree, dict) == False: # Which means that is a leaf node.
        return 1
    num_leaves = 0
    for key in myTree:
        num_leaves = num_leaves + getNumLeaves_basic(myTree[key])
    return num_leaves

###############################################

#print(getNumLeaves_basic(retrieveTree(0)))
#print(getNumLeaves_basic(retrieveTree(1)))

def plotNode_simple(nodeTxt, centerPt, parentPt, nodeType):
    createTreePlot_simple.ax1.annotate(nodeTxt, xy=parentPt, xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)


def plotTree_simple(myTree, centerPt, parentPt):
    if isinstance(myTree, dict) == False:
        plotNode_simple(myTree, centerPt, parentPt, leafNode)
        return
    featureLabel = list(myTree.keys())[0]
    plotNode_simple(featureLabel, centerPt, parentPt, decisionNode)
    featureValDict = myTree[featureLabel]
    i = 0
    for featureValue in featureValDict:
        createTreePlot_simple.ax1.text(centerPt[0] + 0.1*i, centerPt[1] - 0.1, featureValue) 
        # ^^ Putting the label values on the center of the arrows.
        plotTree_simple(featureValDict[featureValue], (centerPt[0] + 0.2*i, centerPt[1] - 0.2), centerPt)
        i = i + 1

"""
The structure of the tree dict is as follows => 
{(Feature Label to split on) -> {Contains all possible values of that label : {Feature Label to split on}}}
So, at the beginning, we have a dict, with only one key, containing the first label using which the dataset
is split. Using this key, we get another dict, which in its keys, contains all the possible values
of this feature. And then repeat. 
"""
        
def createTreePlot_simple(myTree):
    fig = plt.figure()
    fig.clf()
    axprops = dict(xticks=[], yticks=[]) # To remove the coordinate axes.
    createTreePlot_simple.ax1 = plt.subplot(1, 1, 1, frameon=False, **axprops)
    plotTree_simple(myTree, (0.0, 1.0), (0.0, 1.0))
    plt.show()

##########################################################

def getNumLeaves(myTree):
    if isinstance(myTree, dict) == False:
        return 1
    featureLabel = list(myTree.keys())[0]
    featureValDict = myTree[featureLabel]
    num_leaves = 0
    for featureValue in featureValDict:
        num_leaves = num_leaves + getNumLeaves(featureValDict[featureValue])
    return num_leaves

#print(getNumLeaves(retrieveTree(0)))
#print(getNumLeaves(retrieveTree(1)))

def getTreeDepth(myTree):
    if isinstance(myTree, dict) == False:
        return 0
    maxDepth = 0
    featureLabel = list(myTree.keys())[0]
    featureValDict = myTree[featureLabel]
    for featureValue in featureValDict:
        maxDepth = max(maxDepth, 1 + getTreeDepth(featureValDict[featureValue]))
    return maxDepth

#print(getTreeDepth(retrieveTree(0)))
#print(getTreeDepth(retrieveTree(1)))

def plotNode(nodeTxt, centerPt, parentPt, nodeType):
    createPlot.ax1.annotate(nodeTxt, xy=parentPt, xytext=centerPt, textcoords='axes fraction',
                            va='center', ha='center', bbox=nodeType, arrowprops=arrow_args)

def plotMidText(centerPt, parentPt, featVal):
    xMid = centerPt[0] + (parentPt[0] - centerPt[0]) / 2.0
    yMid = centerPt[1] + (parentPt[1] - centerPt[1]) / 2.0
    createPlot.ax1.text(xMid, yMid, featVal)
    
def plotTree(myTree, parentPt, nodeText):
    numLeaves = getNumLeaves(myTree)
    featureLabel = list(myTree.keys())[0]
    centerPt = (plotTree.xOff + (((1.0 + float(numLeaves)) / 2.0) / plotTree.totalW), plotTree.yOff)
    plotMidText(centerPt, parentPt, nodeText)
    plotNode(featureLabel, centerPt, parentPt, decisionNode)
    featureValDict = myTree[featureLabel]
    plotTree.yOff = plotTree.yOff - 1.0 / plotTree.totalD ##
    for featureVal in featureValDict:
        if isinstance(featureValDict[featureVal], dict):
            plotTree(featureValDict[featureVal], centerPt, featureVal)
        else:
            plotTree.xOff = plotTree.xOff + 1.0/plotTree.totalW
            plotNode(featureValDict[featureVal], (plotTree.xOff, plotTree.yOff), centerPt, leafNode)
            plotMidText((plotTree.xOff, plotTree.yOff), centerPt, featureVal)
    plotTree.yOff = plotTree.yOff + 1.0/plotTree.totalD ## To restore the value of yOff.
    

def createPlot(inTree):
    #fig = plt.figure(1, facecolor='white')
    fig = plt.figure()
    fig.clf()
    axprops = dict(xticks=[], yticks=[]) # To remove the coordinate axes.
    createPlot.ax1 = plt.subplot(111, frameon=False, **axprops)
    plotTree.totalW = float(getNumLeaves(inTree))
    plotTree.totalD = float(getTreeDepth(inTree))
    plotTree.xOff = -0.5/plotTree.totalW
    plotTree.yOff = 1.0
    plotTree(inTree, (0.5, 1.0), '')
    plt.show()

"""    
createTreePlot_simple(retrieveTree(0))
createPlot(retrieveTree(0))

createTreePlot_simple(retrieveTree(1))
createPlot(retrieveTree(1))

createTreePlot_simple(retrieveTree(2))
createPlot(retrieveTree(2))
"""