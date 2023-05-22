# -*- coding: utf-8 -*-
"""
Created on Mon Oct 17 19:21:46 2022

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
hidden_size=20 #no. of nodes in hidden layer
input_size=13 #number of features for 1 time point
output_size=3 #single output for age
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
train_dataset, test_dataset = random_split(dataset, [42243, 10561], generator=torch.Generator().manual_seed(66))


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
    model=RNN(input_size=input_size, output_size=output_size, hidden_size=hidden_size, num_layers=num_layers)
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
            #X_train=X_train.to(device)
            y_train=y_train.to(device)
            out, hidden = model(X_train)
            out=torch.reshape(out,(len(out),sequence_length))
            #out=torch.reshape(out,(batch_size,sequence_length))
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
        actual_age1, actual_age2, actual_age3=[], [], []
        predicted_age1, predicted_age2, predicted_age3=[], [], []
        for i, (features, age) in enumerate(test_loader):
            features=features.reshape(-1, sequence_length, input_size).to(device)
            age=age.to(device)
            output,hidden=model(features)
            output=torch.reshape(output,(len(output),sequence_length))
            mae+=(abs(age-output))
            mse+=((age-output)**2)
            if (i+1)%200==0:
                actual_age1.append(age[0][0])
                actual_age2.append(age[0][1])
                actual_age3.append(age[0][2])
                predicted_age1.append(output[0][0])
                predicted_age2.append(output[0][1])
                predicted_age3.append(output[0][2])
            
    
                
        mae = torch.sum(mae/len(test_loader))/sequence_length
        rmse = torch.sum((mse/len(test_loader))**0.5)/sequence_length
        print(f'Mean absolute error of the model on the 10,561 test samples: {mae}')
        print(f'Root mean squared error of the model on the 10,561 test samples: {rmse}')

    #plot loss
    train_plot=plt.figure()
    
    plt.plot(epochi,loser)
    
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    
    plt.show()

    #plot test accuracy
    test_plot1=plt.figure()
    
    test_sample=np.arange(1,len(actual_age1)+1)
    plt.plot(test_sample,actual_age1,c='r',label='Actual age1')
    plt.plot(test_sample,predicted_age1,c='b',label='Predicted age1')
    plt.legend(bbox_to_anchor =(1.35,-0.10), loc='lower right')
    plt.tight_layout()
    plt.title("Test Accuracy")
    plt.xlabel("Test sample")
    plt.ylabel("Age")
    
    
    test_plot2=plt.figure()
    plt.plot(test_sample,actual_age2,c='r',label='Actual age2')
    plt.plot(test_sample,predicted_age2,c='b',label='Predicted age2')
    plt.legend(bbox_to_anchor =(1.35,-0.10), loc='lower right')
    plt.tight_layout()
    plt.title("Test Accuracy")
    plt.xlabel("Test sample")
    plt.ylabel("Age")
    
    
    test_plot3=plt.figure()
    plt.plot(test_sample,actual_age3,c='r',label='Actual age3')
    plt.plot(test_sample,predicted_age3,c='b',label='Predicted age3')
    
    plt.legend(bbox_to_anchor =(1.35,-0.10), loc='lower right')
    plt.tight_layout()
        
    plt.title("Test Accuracy")
    plt.xlabel("Test sample")
    plt.ylabel("Age")
    
    
    plt.show()
    
    """
    rnn=nn.RNN(input_size, hidden_size, num_layers)
    input=X_train #tensor of shape sequence length by input_size
    h0=torch.randn(1,hidden_size)
    output, hn=rnn(input,h0)
    """



