# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:59:16 2022

@author: Agapov Dmitriy
"""
# матрицы Джонса определены в соответствии со статьей DOI: 10.1103/PhysRevE.74.056607

import numpy as np
import random 
import GPNN
from dataConvertor import dataConventor,dataBatching



model = GPNN.GPNN()
model = model.float()
testData = np.load("datasets/datasetNoNoise_LAA_train.npy",allow_pickle=True)
trainData = np.load("datasets/datasetNoNoise_LAA_test.npy",allow_pickle=True)
trainData = dataBatching(dataConventor(trainData),500)
testData = dataBatching(dataConventor(testData),500)
GPNN.trainGPNN(model,trainData,testData,num_epochs = 40)




