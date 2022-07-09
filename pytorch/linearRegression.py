#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jun 17 08:53:06 2022

@author: saad
"""

import torch
import torch.nn as nn
import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

x_numpy, y_numpy = datasets.make_regression(n_samples=100, n_features=1,noise=20, random_state=1)

X= torch.from_numpy(x_numpy.astype(np.float32))
y= torch.from_numpy(y_numpy.astype(np.float32))

#print(y)
y=y.view(y.shape[0],1)
#print(y)

n_sample, n_features= X.shape


input_size= n_features
output_size= 1
model=nn.Linear(input_size,output_size)

learning_rate=0.01
criterion= nn.MSELoss()
optimizer = torch.optim.SGD(model.parameters(),lr=learning_rate)

num_epochs=100

for epoch in range(num_epochs):
    
    y_predicted= model(X)
    loss = criterion(y_predicted,y)
    
    loss.backward()
    
    optimizer.step()
    
    optimizer.zero_grad()
    
    if (epoch+1)%10 == 0:
        print(f'epoch: {epoch+1}, loss= {loss.item():.5f}')
        
predicted = model(X).detach().numpy()
plt.plot(x_numpy, y_numpy, 'ro')
plt.plot(x_numpy, predicted, 'b')
plt.show()