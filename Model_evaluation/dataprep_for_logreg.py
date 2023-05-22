# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 12:57:50 2023

@author: chength
"""

import pandas as pd

df=pd.read_csv('fall_fracture_pred_age.csv').drop(['Unnamed: 0'], axis=1)
columns=['Gender', 'Alcohol', 'Ethnicity', 'Smoking', 'Exercise', 'Fracture', 'Predicted_age']
tp1=df.loc[:, ['X31.0.0', 'X1558.0.0', 'X20116.0.0', 'X21000.0.0', 'X22036.0.0', 'X3005.0.0', 'predicted1']]
tp1.columns=columns
tp2=df.loc[:, ['X31.1.0', 'X1558.1.0', 'X20116.1.0', 'X21000.1.0', 'X22036.1.0', 'X3005.1.0', 'predicted2']]
tp2.columns=columns
tp3=df.loc[:, ['X31.2.0', 'X1558.2.0', 'X20116.2.0', 'X21000.2.0', 'X22036.2.0', 'X3005.2.0', 'predicted3']]
tp3.columns=columns
predicted = pd.concat((tp1,tp2,tp3), axis=0)

predicted.to_csv('fall_fracture_logreg.csv', index=False)