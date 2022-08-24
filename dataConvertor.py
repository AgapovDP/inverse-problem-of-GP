#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  5 14:20:35 2022

@author: dmitryagapov
"""

from torch import from_numpy, split, tensor
import numpy as np



def dataConvertor(data):
    inputData = []
    outputData = []
    label = []
    for i in data:
        inputData.append(i[0])
        vector = [i[1][0,0].real,i[1][1,0].real,i[1][0,1].real,\
                  i[1][1,1].real, i[1][0,0].imag,i[1][0,1].imag,\
                      i[1][1,0].imag,i[1][1,1].imag]
        outputData.append(vector)
        if list(i[2]) == [0,0,0,0]: label.append(0)
        if list(i[2]) == [0,0,0,1]: label.append(1)
        if list(i[2]) == [0,0,1,0]: label.append(2)
        if list(i[2]) == [0,1,0,0]: label.append(3)
        if list(i[2]) == [1,0,0,0]: label.append(4)
        if list(i[2]) == [0,0,1,1]: label.append(5)
        if list(i[2]) == [0,1,0,1]: label.append(6)
        if list(i[2]) == [1,0,0,1]: label.append(7)
        if list(i[2]) == [0,1,1,0]: label.append(8)
        if list(i[2]) == [1,0,1,0]: label.append(9)
        if list(i[2]) == [1,1,0,0]: label.append(10)
        if list(i[2]) == [0,1,1,1]: label.append(11)
        if list(i[2]) == [1,0,1,1]: label.append(12)
        if list(i[2]) == [1,1,0,1]: label.append(13)
        if list(i[2]) == [1,1,1,0]: label.append(14)
        if list(i[2]) == [1,1,1,1]: label.append(15)
        
    return tensor(inputData),tensor(outputData),tensor(label)


def dataBatching(data,batch_len=1):
    inputData =  split(data[0],batch_len)
    outputData = split(data[1],batch_len)
    labels = split(data[2],batch_len)
    batchedData = []
    for i in range(len(inputData)):
        batchedData.append([inputData[i],outputData[i],labels[i]])
    return batchedData
 




