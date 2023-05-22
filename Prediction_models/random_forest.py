# -*- coding: utf-8 -*-
"""
Created on Sun Feb  5 22:29:08 2023

@author: Daniel
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch 

import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
from torch.utils.data import random_split

from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

class UKB(Dataset):

    def __init__(self):
        # Initialize data, download, etc.
        # read with numpy or pandas
        xy = np.loadtxt("ensemble_scaled.csv", delimiter=',', dtype=np.float32, skiprows=1)
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
train_dataset, test_dataset = random_split(dataset, [32886, 8222], generator=torch.Generator().manual_seed(66))

train_index=train_dataset.indices
test_index=test_dataset.indices

df=pd.read_csv('ensemble_scaled.csv').drop(['Unnamed: 0'],axis=1)
training_dataset=df.iloc[train_index]
X_train=training_dataset.drop(['X21022.0.0'],axis=1)
y_train=training_dataset['X21022.0.0']
testing_dataset=df.iloc[test_index]
X_test=testing_dataset.drop(['X21022.0.0'],axis=1)
y_test=testing_dataset['X21022.0.0']

def calculate_metrics(df):
    return {'mae' : mean_absolute_error(df['X21022.0.0'], df.prediction),
            'rmse' : mean_squared_error(df['X21022.0.0'], df.prediction) ** 0.5,
            'r2' : r2_score(df['X21022.0.0'], df.prediction)}


def build_baseline_model(X_train, y_train, X_test, y_test):
    model = RandomForestRegressor(max_depth=5, random_state=6)
    model.fit(X_train, y_train)
    prediction = model.predict(X_test)

    result = pd.DataFrame(y_test)
    result["prediction"] = prediction
    result = result.sort_index()

    return result

df_baseline = build_baseline_model(X_train, y_train, X_test, y_test)
baseline_metrics = calculate_metrics(df_baseline)
print(baseline_metrics)

#Plot prediction accuracy
predict=df_baseline.prediction[::100]
y_test=df_baseline['X21022.0.0'][::100]
n_samples=np.arange(1,len(y_test)+1)
plt.plot(n_samples,predict,c='b',label='Predicted age')
plt.plot(n_samples,y_test,c='r',label='Actual age')
plt.title("Test Accuracy")
plt.xlabel("Test sample")
plt.ylabel("Age")
plt.legend()
plt.show()