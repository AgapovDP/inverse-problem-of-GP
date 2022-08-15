# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:59:16 2022

@author: Agapov Dmitriy
"""


import torch.nn as nn
from dataConvertor import dataConventor,dataBatching
import numpy as np

standartSeq = nn.Sequential(
     nn.Linear(3, 256),
     nn.Unflatten(1, (1,16, 16)),
     nn.ReLU(),
     nn.Conv2d(1, 20, 2, padding=2),
     nn.ReLU(),
     nn.Conv2d(20, 50, 2, padding=2),
     nn.Flatten(),
     nn.ReLU(),
     nn.Linear(24200, 7)

)


class GPNN(nn.Module):
    def __init__(self, Seq = standartSeq ):
        super().__init__()
        self.layers_stack = Seq
  

    def forward(self, x):
        x = self.layers_stack(x)
        return x
    

import torch
import torch.optim as optim

model = GPNN()
model = model.float()

def trainGPNN(model, trainloader,testloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 20
    loss_hist = [] # for plotting
    val_arr_test = []
    val_arr_train = []
    for epoch in range(num_epochs):
        hist_loss = 0
        numBatch = 0
        while numBatch < len(trainloader):
            value, labels = trainloader[numBatch]
            numBatch = numBatch + 1
            optimizer.zero_grad()
            Y_pred = model(value.float())
            loss = criterion(Y_pred, labels.float())
            loss.backward()    
            optimizer.step()
            hist_loss += loss.item()
        loss_hist.append(hist_loss /len(trainloader))
        #val_arr_train.append(validate(model,trainloader,device))
        #val_arr_test.append(validate(model,testloader,device))
        print(f"Epoch={epoch} loss={loss_hist[epoch]:.4f}")
    return loss_hist

def validate(model,data):
    correct = 0
    total = 0
    numBatch = 0
    with torch.no_grad():
        while numBatch < len(data):
            value, labels = data[numBatch]
            labels = labels.float()
            numBatch = numBatch + 1
            outputs = model(value.float())
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.float()).sum().item()

    return correct / total 


testData = np.load("testLAA_test.npy",allow_pickle=True)
trainData = np.load("testLAA_train.npy",allow_pickle=True)
trainData = dataBatching(dataConventor(trainData),100)
testData = dataBatching(dataConventor(testData),100)
trainGPNN(model,trainData,testData)
