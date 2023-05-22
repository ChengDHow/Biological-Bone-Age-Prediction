# -*- coding: utf-8 -*-
"""
Created on Sun Feb 12 17:03:45 2023

@author: Daniel
"""

import pandas as pd
import numpy as np

#Select desired variables, confounders and target
df=pd.read_csv('ukb52305.csv', usecols=['21003-0.0','21003-1.0','21003-2.0',
                                        '4100-0.0','4100-1.0','4100-2.0',
                                         '4119-0.0','4119-1.0','4119-2.0',
                                         '4103-0.0','4103-1.0','4103-2.0',
                                         '4122-0.0','4122-1.0','4122-2.0',
                                         '4101-0.0','4101-1.0','4101-2.0',
                                         '4120-0.0','4120-1.0','4120-2.0',
                                         '4105-0.0','4105-1.0','4105-2.0',
                                         '4124-0.0','4124-1.0','4124-2.0',
                                         '20116-0.0','20116-1.0','20116-2.0',
                                         '1558-0.0','1558-1.0','1558-2.0',
                                         '31-0.0', '21000-0.0','22036-0.0'])

df.to_csv('21003.csv', index = False)


#Filter out rows where independent variables are all nan (also for target variable)
colnames=df.columns
df_array=np.array(df)

temp=[]
for i in df_array:
    if np.isnan(i[4:28]).all():
        continue
    elif np.isnan(i[33:35]).all():
        continue
    else:
        temp.append(i)

filtered=pd.DataFrame(temp)
filtered.columns=colnames

filtered.to_csv('filtered_21003.csv', index = False)


#Calculate missingness of each column
colnames=filtered.columns

for col in colnames:
    num_na=filtered[col].isna().sum()
    print (num_na/52804)
    

#Create ground truth dataset    
colnames=filtered.columns
df_array=np.array(filtered)

temp=[]
for i in df_array:
    if np.isnan(i[4:28]).any():
        continue
    elif np.isnan(i[33:35]).any():
        continue
    else:
        temp.append(i)

ground=pd.DataFrame(temp)
ground.columns=colnames

ground.to_csv('ground_21003.csv', index = False)





