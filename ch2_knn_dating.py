#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:52:17 2018

@author: lakshay
"""

import numpy as np
import operator

#filename = "./data/datingTestSet.txt"

def file2matrix(filename):
    f = open(filename, 'r')

    lines = f.readlines()
    X = []
    y = []
    for line in lines:
        line = line.strip()
        elements = line.split('\t')
        X.append(elements[0:3])
        y.append(elements[3])
    
    X = np.array(X)
    return X, y
