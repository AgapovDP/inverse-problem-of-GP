# -*- coding: utf-8 -*-
"""
Created on Tue Jul 26 23:59:16 2022

@author: Agapov Dmitriy
"""


import torch.nn as nn


standartSeq = nn.Sequential(
     nn.Linear(8, 256),
     nn.Unflatten(1, (1,16, 16)),
     nn.ReLU(),
     nn.Conv2d(1, 20, 2, padding=2),
     nn.ReLU(),
     nn.Conv2d(20, 50, 2, padding=2),
     nn.Flatten(),
     nn.ReLU(),
     nn.Linear(24200, 8)

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

def trainGPNN(trainloader,testloader):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print('Using device:', device)
    model = GPNN().to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    num_epochs = 2
    loss_hist = [] # for plotting
    val_arr_test = []
    val_arr_train = []
    for epoch in range(num_epochs):
        hist_loss = 0
        for _, batch in enumerate(trainloader, 0): # get batch
            # parse batch 
            imgs, labels = batch
            imgs, labels = imgs.to(device), labels.to(device)
            # sets the gradients of all optimized tensors to zero.
            optimizer.zero_grad() 
            # get outputs
            Y_pred = model(imgs) 
            # calculate loss
            loss = criterion(Y_pred, labels)
            # calculate gradients
            loss.backward() 
            # performs a single optimization step (parameter update)
        optimizer.step()
        hist_loss += loss.item()
        loss_hist.append(hist_loss /len(trainloader))
        val_arr_train.append(validate(model,trainloader,device))
        val_arr_test.append(validate(model,testloader,device))
        print(f"Epoch={epoch} loss={loss_hist[epoch]:.4f}")

def validate(model,testloader,device):
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in testloader:
            outputs = model(images.to(device))
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels.to(device)).sum().item()

    return correct / total 