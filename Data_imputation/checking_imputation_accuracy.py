# -*- coding: utf-8 -*-
"""
Created on Mon Jan 30 18:06:45 2023

@author: chength
"""

import pandas as pd
import numpy as np

df_amputated=pd.read_csv('ensemble_amputated.csv').drop(['Unnamed: 0', '21022-0.0'], axis=1)
df_truth=pd.read_csv('ensemble_ground_truth.csv', usecols=['4100-0.0', '4100-1.0', '4100-2.0', '4101-0.0', '4101-1.0', '4101-2.0',
                                                           '4103-0.0', '4103-1.0', '4103-2.0', '4105-0.0', '4105-1.0', '4105-2.0', 
                                                           '4119-0.0', '4119-1.0', '4119-2.0', '4120-0.0', '4120-1.0', '4120-2.0', 
                                                           '4122-0.0', '4122-1.0', '4122-2.0', '4124-0.0', '4124-1.0', '4124-2.0', 
                                                           '23200-2.0', '23200-3.0', '23203-2.0', '23203-3.0', 
                                                           '23204-2.0', '23204-3.0', '23212-2.0', '23212-3.0', '23224-2.0', 
                                                           '23224-3.0', '23225-2.0', '23225-3.0', '23226-2.0', '23226-3.0', 
                                                           '23230-2.0', '23230-3.0', '23231-2.0', '23231-3.0', '23232-2.0', 
                                                           '23232-3.0', '23233-2.0', '23233-3.0', '23234-2.0', '23234-3.0', 
                                                           '23240-2.0', '23240-3.0', '23241-2.0', '23241-3.0', '23291-2.0', 
                                                           '23291-3.0', '23304-2.0', '23304-3.0', '23305-2.0', '23305-3.0', 
                                                           '23306-2.0', '23306-3.0', '23307-2.0', '23307-3.0', '23308-2.0', 
                                                           '23308-3.0', '23309-2.0', '23309-3.0', '23310-2.0', '23310-3.0', 
                                                           '23311-2.0', '23311-3.0', '23312-2.0', '23312-3.0', '23317-2.0', 
                                                           '23317-3.0', '23318-2.0', '23318-3.0'])
df_lasso=pd.read_csv('ensemble_lasso_smol.csv')

index=[]
amputated_array=np.array(df_amputated)
for i in range (len(amputated_array)):
    for j in range (len(amputated_array[0])):
        if np.isnan(amputated_array[i][j]):
            index.append([i,j])

truth_array=np.array(df_truth)
lasso_array=np.array(df_lasso)
mae=0
for i in index:
    mae+=abs((truth_array[i[0]][i[1]]-lasso_array[i[0]][i[1]])/truth_array[i[0]][i[1]])
mae=mae/len(index)

errors=[]
for i in range(len(amputated_array[0])):
    temp=amputated_array[:,i]
    actual=truth_array[:,i]
    forecast=lasso_array[:,i]
    mape=0
    n=0
    for j in range(len(temp)):
        if np.isnan(temp[j]):
            mape+=abs((actual[j]-forecast[j])/actual[j])
            n+=1
        else:
            continue
    mape=mape/n
    errors.append(mape)

x=np.array(errors[:24]).reshape((3,8),order='F')
y=np.array(errors[24:]).reshape((2,26),order='F')


bmd_truth = df_truth['4124-0.0']
bmd_lasso = df_lasso['X4124.0.0']
truth_y = []
lasso_y = []
for i in range(len(bmd_lasso)):
    if bmd_lasso[i] == bmd_truth[i]:
        continue
    else:
        truth_y.append(bmd_truth[i])
        lasso_y.append(bmd_lasso[i])
        
x = np.arange(1,79)

import matplotlib.pyplot as plt
plt.plot(x, truth_y, label='Ground truth', c='r')
plt.plot(x, lasso_y, label='Imputed value', c='b')

plt.title("Imputation accuracy of variable with 70% missing data")
plt.xlabel("Sample Number")
plt.ylabel("Bone Mineral Density")
plt.legend(loc='upper right')

plt.show()

mape=0
for i in range(len(truth_y)):
    mape+=abs((truth_y[i]-lasso_y[i])/truth_y[i])
mape=1-(mape/len(truth_y))








            