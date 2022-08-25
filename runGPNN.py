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
import torch
import time 

classSeq_v3 = nn.Sequential(
     nn.Linear(9, 400),
     nn.Unflatten(1, (1,20, 20)),
     nn.ReLU(),
     nn.Conv2d(1, 10, 4, padding=2),
     nn.ReLU(),
     nn.Conv2d(10, 30, 4, padding=2),
     nn.Flatten(),
     nn.ReLU(),
     nn.Dropout(0.3),
     nn.Linear(14520, 2000),
     nn.ReLU(),
     nn.Dropout(0.3),
     nn.Linear(2000, 100),
     nn.ReLU(),
     nn.Linear(100, 16) 

)

testData = np.load("datasets/datasetNoNoise_randomObject_train_250000.npy",allow_pickle=True)
trainData = np.load("datasets/datasetNoNoise_randomObject_test_250000.npy",allow_pickle=True)
trainData = dataBatching(dataConvertor(trainData),1000)
testData = dataBatching(dataConvertor(testData),1000)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Using device:', device)
model = GPNN.GPNN().to(device)
model = model.float()

time_0 = time.time()
manual_seed(42)
random.seed(42)
np.random.seed(42)
loss, trainVal, testVal, acc_train, acc_test = GPNN.trainGPNN(model,trainData,\
                       testData,num_epochs = 21,criterion = nn.CrossEntropyLoss,\
                           learning_rate = 0.001, device = device, maxBatch = 20)
print("Duration:", time.time()-time_0)
plt.plot(acc_train)
plt.plot(acc_test)
    
# просто промежуточная функция для формирования графиков. Удалю попозже
def somePlot(data,num_epochs):
    localData = [[],[],[],[],[],[],[],[]]
    for j in range(8):
        for i in range(num_epochs):
            localData[j].append(data[i][j])
    return localData