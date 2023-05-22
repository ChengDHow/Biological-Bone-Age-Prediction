# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 10:56:00 2023

@author: chength
"""

from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np

df=pd.read_csv('fracture_onetp_imputed.csv')
df['X31.1.0']=df['X31.0.0']
df['X31.2.0']=df['X31.0.0']
df['X21000.1.0']=df['X21000.0.0']
df['X21000.2.0']=df['X21000.0.0']
df['X22036.1.0']=df['X22036.0.0']
df['X22036.2.0']=df['X22036.0.0']



cols=['X21003.0.0', 
      'X31.0.0', 'X1558.0.0', 'X4100.0.0', 'X4101.0.0', 'X4103.0.0', 'X4105.0.0', 'X4119.0.0', 'X4120.0.0', 'X4122.0.0', 'X4124.0.0', 'X20116.0.0', 'X21000.0.0', 'X22036.0.0', 
      'X31.1.0', 'X1558.1.0', 'X4100.1.0', 'X4101.1.0', 'X4103.1.0', 'X4105.1.0', 'X4119.1.0', 'X4120.1.0', 'X4122.1.0', 'X4124.1.0', 'X20116.1.0', 'X21000.1.0', 'X22036.1.0', 
      'X31.2.0', 'X1558.2.0', 'X4100.2.0', 'X4101.2.0', 'X4103.2.0', 'X4105.2.0', 'X4119.2.0', 'X4120.2.0', 'X4122.2.0', 'X4124.2.0', 'X20116.2.0', 'X21000.2.0', 'X22036.2.0',
      'X3005.0.0']

df=df[cols]


colname=df.columns

scaler=MinMaxScaler(feature_range=(-1,1))
df[colname[1:40]]=scaler.fit_transform(df[colname[1:40]])

df.to_csv('fracture_onetp_scaled.csv', index=False)