# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:59:16 2022

@author: Agapov Dmitriy
"""
# матрицы Джонса определены в соответствии со статьей DOI: 10.1103/PhysRevE.74.056607

import numpy as np
import random 
import GPNN
from dataConvertor import dataConvertor,dataBatching
import matplotlib.pyplot as plt


model = GPNN.GPNN()
model = model.float()
testData = np.load("datasets/datasetNoNoise_randomObject_train.npy",allow_pickle=True)
trainData = np.load("datasets/datasetNoNoise_randomObject_test.npy",allow_pickle=True)
trainData = dataBatching(dataConvertor(trainData),150)
testData = dataBatching(dataConvertor(testData),150)
loss, trainVal, testVal = GPNN.trainGPNN(model,trainData,testData,num_epochs = 15)


# просто промежуточная функция для формирования графиков. Удалю попозже
def somePlot(data,num_epochs):
    localData = [[],[],[],[],[],[],[],[]]
    for j in range(8):
        for i in range(num_epochs):
            localData[j].append(data[i][j])
    return localData