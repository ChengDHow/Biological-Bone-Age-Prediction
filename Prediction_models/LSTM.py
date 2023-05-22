# -*- coding: utf-8 -*-
"""
Created on Fri Dec  9 16:01:47 2022

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

#Set hyperparameters
hidden_size=3 #no. of nodes in hidden layer
input_size=13 #number of features for 1 time point
output_size=1 #single output for age
sequence_length=3 #number of time points
num_layers=1
batch_size=32 #must be >=1 or <=number of training samples in the training dataset
#number of iterations=number of samples divided by batch size --> for 1 epoch


#Preparing training and testing datasets, can reshape beforehand so that just have to read and convert to tensor
# X needs to be: (batch_size, sequence_length, input_size)
class UKB(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt("3tp_scaled(-1 to 1).csv", delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # Skip the first column which is just indexing. Second column is age and the rest are features.
        self.x_data = torch.from_numpy(xy[:, 2:]) # size [n_samples, n_features]
        self.y_data = torch.from_numpy(xy[:, [1]]) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x_data[index], self.y_data[index]

    # call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset=UKB()
train_dataset, test_dataset = random_split(dataset, [158794, 39699], generator=torch.Generator().manual_seed(66))


#Data Loader
train_loader=torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


#Building of RNN 
class LSTM(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(LSTM, self).__init__()
        
        #defining parameters
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        
        #defining layers
        self.lstm=nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        #fully connected layer
        self.fc=nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size=x.size(0) #first dimension of the tensor
        
        #initialize hidden state and cell state
        h0=self.init_hidden(batch_size).to(device)
        c0=self.init_cell(batch_size).to(device)
        
        #passing input and hidden state into the model and obtaining output
        out, hidden=self.lstm(x, (h0,c0))
        #out: batch_size, sequence_length, hidden_size
        
        #reshape output to fit it into the fully connected layer 
        #out = out.contiguous().view(-1, self.hidden_size)
        out=out[:, -1, :]
        #tensor size (n, hidden_size)
        
        out = self.fc(out)
        #tensor size (n, output_size)
        
        return out, hidden

    def init_hidden(self, batch_size):
        #generate initial hidden state
        hidden=torch.randn(self.num_layers, batch_size, self.hidden_size)
        return hidden
    
    def init_cell(self, batch_size):
        #generate initial hidden state
        cell=torch.randn(self.num_layers, batch_size, self.hidden_size)
        return cell

if __name__ == "__main__":
    
    #Instantiate the model with hyperparameters
    model=LSTM(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers)
    model.to(device)
    
    #Define hyperparameters
    n_epochs=100
    lr=0.0003 #best learning rate for adam, hands down
    
    #Define loss function and optimizer
    criterion=nn.L1Loss()
    optimizer=torch.optim.Adam(model.parameters(), lr=lr)
    
    #Training run
    loser=[]
    epochi=[]
    training_progress=len(train_loader) 
    for epoch in range(1, n_epochs+1):
        for i, (X_train, y_train) in enumerate(train_loader):
            optimizer.zero_grad() # Clears existing gradients from previous epoch
            X_train=X_train.reshape(-1, sequence_length, input_size).to(device)
            y_train=y_train.to(device)
            out, hidden = model(X_train)
            #loss = criterion(out, y_train.view(-1).long())
            loss = criterion(out, y_train)
            loss.backward() # Does backpropagation and calculates gradients
            optimizer.step() # Updates the weights accordingly
            
            if (i+1)%100 == 0:
                print (f'Epoch [{epoch}/{n_epochs}], Progress [{i+1}/{training_progress}], Loss: {loss.item():.4f}')
        loser.append(loss.item())
        epochi.append(epoch)

            
    #Testing run
    with torch.no_grad():
        mae=0
        mse=0
        actual_age=[]
        predicted_age=[]
        for i, (features, age) in enumerate(test_loader):
            features=features.reshape(-1, sequence_length, input_size).to(device)
            age=age.to(device)
            output,hidden=model(features)
            mae+=(abs(age-output))
            mse+=((age-output)**2)
            if (i+1)%1000==0:
                actual_age.append(age)
                predicted_age.append(output)
    
                
        mae = mae/len(test_loader)
        rmse = (mse/len(test_loader))**0.5
        print(f'Mean absolute error of the model on the 39,699 test samples: {mae}')
        print(f'Root mean squared error of the model on the 39,699 test samples: {rmse}')
    
    
    #plot loss
    train_plot=plt.figure()
    
    plt.plot(epochi,loser)
    
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.show()
    
    #plot test accuracy
    test_plot=plt.figure()
    
    test_sample=np.arange(1,len(actual_age)+1)
    plt.plot(test_sample,actual_age,c='r',label='Actual age')
    plt.plot(test_sample,predicted_age,c='b',label='Predicted age')
    plt.legend()
     
    
    plt.title("Test Accuracy")
    plt.xlabel("Test sample")
    plt.ylabel("Age")
    
    
    plt.show()    