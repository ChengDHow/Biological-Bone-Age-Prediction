# -*- coding: utf-8 -*-
"""
Created on Thu Dec 29 14:50:56 2022

@author: Daniel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split


#Set device to GPU (or CPU if GPU not available)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')


#Preparing training and testing datasets, can reshape beforehand so that just have to read and convert to tensor
# X needs to be: (batch_size, sequence_length, input_size)
class UKB(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt("21003_scaled.csv", delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # Skip the first column which is just indexing. Second column is age and the rest are features.
        self.x_data = torch.from_numpy(xy[:, 4:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, 1:4]) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset=UKB()
train_dataset, test_dataset = random_split(dataset, [42243, 10561], generator=torch.Generator().manual_seed(60))


#Data Loader
train_loader=torch.utils.data.DataLoader(dataset=train_dataset, batch_size=1, shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)

train_list=[]
for X_train, y_train in train_loader:
    train_list.append(y_train.numpy()[0][0])
train_list=np.array(train_list)
mean_train=np.mean(train_list)
train_25th=np.percentile(train_list,25)
train_50th=np.percentile(train_list,50)
train_75th=np.percentile(train_list,75)
max_train=np.max(train_list)
min_train=np.min(train_list)

train_plot=plt.figure()
    
plt.title("Age distribution for training dataset")
plt.xlabel("Age")
plt.ylabel("Percentage")

plt.hist(train_list, weights=np.ones(len(train_list))/len(train_list))
plt.show()

print('Mean train:', mean_train)
print('25th train:', train_25th)
print('median:', train_50th)
print('75th train:', train_75th)
print('max train:', max_train)
print('min train:', min_train)

test_list=[]
for X_test, y_test in test_loader:
    test_list.append(y_test.numpy()[0][0])
test_list=np.array(test_list)
mean_test=np.mean(test_list)
test_25th=np.percentile(test_list,25)
test_50th=np.percentile(test_list,50)
test_75th=np.percentile(test_list,75)
max_test=np.max(test_list)
min_test=np.min(test_list)

test_plot=plt.figure()
    
plt.title("Age distribution for testing dataset")
plt.xlabel("Age")
plt.ylabel("Percentage")

plt.hist(test_list, weights=np.ones(len(test_list))/len(test_list))
plt.show()

print('Mean test:', mean_test)
print('25th test:', test_25th)
print('median:', test_50th)
print('75th test:', test_75th)
print('max test:', max_test)
print('min test:', min_test)

    

    
    
    