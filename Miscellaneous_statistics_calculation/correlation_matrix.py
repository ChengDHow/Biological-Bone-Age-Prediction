# -*- coding: utf-8 -*-
"""
Created on Thu Dec  8 13:59:29 2022

@author: Daniel
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

df=pd.read_csv('3tp_lasso_reshaped.csv').drop(['Unnamed: 0'],axis=1)
colnames=df.columns
df=df[colnames[1:14]]
df.columns=['Gender','Alcohol','Ankle width (left)','Ultrasound attenuation (left)',
            'Speed of sound (left)','BMD (left)','Ankle width (right)','Ultrasound attenuation (right)',
            'Speed of sound (right)','BMD (right)','Smoking','Ethnicity','Physical activity']
cor_matrix=df.corr()

#plot correlation matrix
f = plt.figure(figsize=(19, 15))
plt.matshow(cor_matrix, fignum=f.number)
plt.xticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14, rotation=45)
plt.yticks(range(df.select_dtypes(['number']).shape[1]), df.select_dtypes(['number']).columns, fontsize=14)
cb = plt.colorbar()
cb.ax.tick_params(labelsize=14)
plt.title('Correlation Matrix', fontsize=16)

