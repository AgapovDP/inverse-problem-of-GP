# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:59:16 2022

@author: Agapov Dmitriy
"""
# матрицы Джонса определены в соответствии со статьей DOI: 10.1103/PhysRevE.74.056607

import numpy as np
import random 
import GPNN_LPA
import matplotlib.pyplot as plt
from torch import nn,manual_seed,from_numpy, split, tensor
import torch
import time 
torch.cuda.empty_cache()


def dataConvertor(data):
    inputData = []
    outputData = []
    for i in data:
        inputData.append(i[0])
        outputData.append(i[1])
    return tensor(inputData),tensor(outputData)


def dataBatching(data,batch_len=1):
    inputData =  split(data[0],batch_len)
    outputData = split(data[1],batch_len)
    batchedData = []
    for i in range(len(inputData)):
        batchedData.append([inputData[i],outputData[i]])
    return batchedData

testData = np.load("datasets/datasetNoNoise_LPA_train_50000.npy",allow_pickle=True)
trainData = np.load("datasets/datasetNoNoise_LPA_test_50000.npy",allow_pickle=True)
trainData = dataBatching(dataConvertor(trainData),10)
testData = dataBatching(dataConvertor(testData),10)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
model = GPNN_LPA.GPNN_LPA().to(device)
model = model.float()

time_0 = time.time()
manual_seed(42)
random.seed(42)
np.random.seed(42)
loss, trainVal, testVal, acc_train, acc_test = GPNN_LPA.trainGPNN(model,trainData,\
                       testData ,num_epochs = 3,\
                           learning_rate = 0.001, device = device, maxBatch = 3)
print("Duration:", time.time()-time_0)
plt.plot(trainVal)
plt.plot(testVal)
    
# просто промежуточная функция для формирования графиков. Удалю попозже
def somePlot(data,num_epochs):
    localData = [[],[],[],[],[],[],[],[]]
    for j in range(8):
        for i in range(num_epochs):
            localData[j].append(data[i][j])
    return localData