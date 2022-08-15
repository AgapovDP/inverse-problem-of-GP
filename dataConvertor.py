#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:20:35 2022

@author: dmitryagapov
"""

from torch import from_numpy, split, tensor
import numpy as np



def dataConventor(data):
    inputData = []
    outputData = []
    for i in data:
        inputData.append(i[0])
        vector = [i[1][0,0].real,i[1][1,0].real,i[1][1,1].real,\
                  i[1][0,0].imag,i[1][0,1].imag,i[1][1,0].imag,\
                      i[1][1,1].imag]
        outputData.append(vector)
    return tensor(inputData),tensor(outputData)


def dataBatching(data,batch_len=1):
    inputData =  split(data[0],batch_len)
    outputData = split(data[1],batch_len)
    batchedData = []
    for i in range(len(inputData)):
        batchedData.append([inputData[i],outputData[i]])
    return batchedData
        
test = np.load("testLAA.npy",allow_pickle=True)
train = dataBatching(dataConventor(test),10)




