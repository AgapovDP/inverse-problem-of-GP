# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:59:16 2022

@author: Agapov Dmitriy
"""


import torch.nn as nn
from dataConvertor import dataConvertor,dataBatching
import numpy as np

standartSeq = nn.Sequential(
     nn.Linear(8, 256),
     nn.Unflatten(1, (1,16, 16)),
     nn.ReLU(),
     nn.Conv2d(1, 15, 3, padding=2),
     nn.ReLU(),
     nn.Conv2d(15, 30, 3, padding=2),
     nn.Flatten(),
     nn.ReLU(),
     nn.Dropout(0.3),
     nn.Linear(12000, 1000),
     nn.ReLU(),
     nn.Linear(1000, 8)  

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


def trainGPNN(model, trainloader,testloader, num_epochs = 2, criterion = nn.MSELoss,\
              optimizer = optim.Adam,learning_rate = 0.001):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    criterion = criterion()
    optimizer = optimizer(model.parameters(), lr=learning_rate)
    num_epochs = num_epochs
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
        val_arr_train.append(validate(model,trainloader))
        val_arr_test.append(validate(model,testloader))
        print(f"Epoch={epoch} loss={loss_hist[epoch]:.5f}")
    return loss_hist,val_arr_train,val_arr_test

def validate(model,data):
    numBatch = 0
    total = 0
    while numBatch < len(data):
        value, labels = data[numBatch]
        numBatch = numBatch + 1
        total += (abs(model(value.float())-labels)**2).mean(0)
    return total/len(data)
    


