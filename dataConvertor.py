#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:20:35 2022

@author: dmitryagapov
"""

from torch import from_numpy
import numpy as np

test = np.load("testLAA.npy",allow_pickle=True)

def dataConventor(data):
    inputData = []
    outputData = []
    for i in test:
        inputData.append(i[0])
        outputData.append(i[1])
    return from_numpy(np.array(inputData)),from_numpy(np.array(outputData))

