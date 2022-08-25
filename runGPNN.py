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
from torch import nn,manual_seed




model = GPNN.GPNN()
model = model.float()
testData = np.load("datasets/datasetNoNoise_randomObject_train_12000.npy",allow_pickle=True)
trainData = np.load("datasets/datasetNoNoise_randomObject_test_12000.npy",allow_pickle=True)
trainData = dataBatching(dataConvertor(trainData),150)
testData = dataBatching(dataConvertor(testData),150)

manual_seed(42)
random.seed(42)
np.random.seed(42)
loss, trainVal, testVal, acc_train, acc_test = GPNN.trainGPNN(model,trainData,\
                       testData,num_epochs = 80,criterion = nn.CrossEntropyLoss,\
                           learning_rate = 0.001)


# просто промежуточная функция для формирования графиков. Удалю попозже
def somePlot(data,num_epochs):
    localData = [[],[],[],[],[],[],[],[]]
    for j in range(8):
        for i in range(num_epochs):
            localData[j].append(data[i][j])
    return localData