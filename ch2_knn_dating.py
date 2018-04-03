#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Apr  3 16:52:17 2018

@author: lakshay
"""

import numpy as np
import operator

input_file = "./data/datingTestSet.txt"
f = open(input_file, 'r')

lines = f.readlines()
X = []
y = []
for line in lines:
    elements = line.split('\t')
    X.append([elements[0], elements[1], elements[2]])
    y.append(elements[3].split('\n')[0])
    
X = np.array(X)
y = np.array(y)