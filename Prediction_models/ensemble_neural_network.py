# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 11:08:54 2023

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
output_size=1 #single output for age
num_layers=1
batch_size=32 #must be >=1 or <=number of training samples in the training dataset
#number of iterations=number of samples divided by batch size --> for 1 epoch

#parameters for each neural network
hidden_size1=20 #no. of nodes in hidden layer
input_size1=13 #number of features for 1 time point
sequence_length1=3 #number of time points

hidden_size2=50
input_size2=26
sequence_length2=2



#Preparing training and testing datasets, can reshape beforehand so that just have to read and convert to tensor
# X needs to be: (batch_size, sequence_length, input_size)
class UKB(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt("ensemble_scaled.csv", delimiter=',', dtype=np.float32, skiprows=1)
        self.n_samples = xy.shape[0]

        # Skip the first column which is just indexing. Second column is age and the rest are features.
        self.x1_data = torch.from_numpy(xy[:, 2:41]) # size [n_samples, n_features with 3 time points]
        self.x2_data = torch.from_numpy(xy[:, 41:]) # size [n_samples, n_features with 2 time points]
        self.y_data = torch.from_numpy(xy[:, [1]]) # size [n_samples, 1]

    # support indexing such that dataset[i] can be used to get i-th sample
    def __getitem__(self, index):
        return self.x1_data[index], self.x2_data[index], self.y_data[index]

    # call len(dataset) to return the size
    def __len__(self):
        return self.n_samples

dataset=UKB()
train_dataset, test_dataset = random_split(dataset, [32886, 8222], generator=torch.Generator().manual_seed(66))


#Data Loader
train_loader=torch.utils.data.DataLoader(dataset=train_dataset, batch_size=batch_size, shuffle=True)

test_loader=torch.utils.data.DataLoader(dataset=test_dataset, batch_size=1, shuffle=False)


#Building of RNN 
class RNN(nn.Module):
    def __init__(self, input_size, output_size, hidden_size, num_layers):
        super(RNN, self).__init__()
        
        #defining parameters
        self.num_layers=num_layers
        self.hidden_size=hidden_size
        
        #defining layers
        self.rnn=nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        #fully connected layer
        self.fc=nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        batch_size=x.size(0) #first dimension of the tensor
        
        #initialize hidden state
        h0=self.init_hidden(batch_size).to(device)
        
        #passing input and hidden state into the model and obtaining output
        out, hidden=self.rnn(x, h0)
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

if __name__ == "__main__":
    
    #Instantiate the model with hyperparameters
    model1=RNN(input_size=input_size1, output_size=output_size, hidden_size=hidden_size1, num_layers=num_layers)
    model2=RNN(input_size=input_size2, output_size=output_size, hidden_size=hidden_size2, num_layers=num_layers)
    model1.to(device)
    model2.to(device)
    
    #Define hyperparameters
    n_epochs=100
    lr=0.0003 #best learning rate for adam, hands down
    
    #Define loss function and optimizer
    criterion=nn.L1Loss()
    optimizer1=torch.optim.Adam(model1.parameters(), lr=lr)
    optimizer2=torch.optim.Adam(model2.parameters(), lr=lr)
    
    #Training run
    loser=[]
    epochi=[]
    training_progress=len(train_loader) 
    for epoch in range(1, n_epochs+1):
        for i, (X_train1, X_train2, y_train) in enumerate(train_loader):
            optimizer1.zero_grad() # Clears existing gradients from previous epoch
            optimizer2.zero_grad() # Clears existing gradients from previous epoch
            X_train1=X_train1.reshape(-1, sequence_length1, input_size1).to(device)
            X_train2=X_train2.reshape(-1, sequence_length2, input_size2).to(device)
            #X_train=X_train.to(device)
            y_train=y_train.to(device)
            out1, hidden1 = model1(X_train1)
            out2, hidden2 = model2(X_train2)
            #out=torch.reshape(out,(len(out),sequence_length))
            #out=torch.reshape(out,(batch_size,sequence_length))
            #loss = criterion(out, y_train.view(-1).long())
            #out=torch.div(torch.add(out1,out2),2)#for mean of outputs
            out=torch.add(torch.multiply(out1,0.8),torch.multiply(out2,0.2))
            loss = criterion(out, y_train)
            loss.backward() # Does backpropagation and calculates gradients
            optimizer1.step() # Updates the weights accordingly
            optimizer2.step() # Updates the weights accordingly
            
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
        for i, (features1, features2, age) in enumerate(test_loader):
            features1=features1.reshape(-1, sequence_length1, input_size1).to(device)
            features2=features2.reshape(-1, sequence_length2, input_size2).to(device)
            age=age.to(device)
            output1,hidden1=model1(features1)
            output2,hidden2=model2(features2)
            #output=torch.reshape(output,(len(output),sequence_length))
            #output=torch.div(torch.add(output1,output2),2)#for mean of outputs
            output=torch.add(torch.multiply(output1,0.8),torch.multiply(output2,0.2))
            mae+=(abs(age-output))
            mse+=((age-output)**2)
            if (i+1)%100==0:
                actual_age.append(age)
                predicted_age.append(output)

    
                
        mae = mae/len(test_loader)
        rmse = (mse/len(test_loader))**0.5
        print(f'Mean absolute error of the model on the 8,222 test samples: {mae}')
        print(f'Root mean squared error of the model on the 8,222 test samples: {rmse}')

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

     
    
    plt.title("Test Accuracy")
    plt.xlabel("Test sample")
    plt.ylabel("Age")
    plt.legend()
    
    
    plt.show()
    
    """
    rnn=nn.RNN(input_size, hidden_size, num_layers)
    input=X_train #tensor of shape sequence length by input_size
    h0=torch.randn(1,hidden_size)
    output, hn=rnn(input,h0)
    """
