# -*- coding: utf-8 -*-
"""
Created on Sat Nov 12 16:23:03 2022

@author: Daniel
"""

import pandas as pd
import numpy as np
import math as math

df_metrics=pd.read_csv('imputation_accuracy.csv').drop(['Unnamed: 0.1'], axis=1)
df_truth=pd.read_csv("ground_truth(3tp).csv", 
                     usecols=['4100-0.0','4100-1.0','4100-2.0',
                                '4101-0.0','4101-1.0','4101-2.0',
                                '4103-0.0','4103-1.0','4103-2.0',
                                '4105-0.0','4105-1.0','4105-2.0',
                                '4119-0.0','4119-1.0','4119-2.0',
                                '4120-0.0','4120-1.0','4120-2.0',
                                '4122-0.0','4122-1.0','4122-2.0',
                                '4124-0.0','4124-1.0','4124-2.0'])
df_amputated=pd.read_csv('3tp_amputated.csv').drop(['Unnamed: 0','21022-0.0'], axis=1)
df_imputed=pd.read_csv('ri.csv').drop(['X21022.0.0'], axis=1)
df_imputed.columns=['4100-0.0','4100-1.0','4100-2.0',
                    '4101-0.0','4101-1.0','4101-2.0',
                    '4103-0.0','4103-1.0','4103-2.0',
                    '4105-0.0','4105-1.0','4105-2.0',
                    '4119-0.0','4119-1.0','4119-2.0',
                    '4120-0.0','4120-1.0','4120-2.0',
                    '4122-0.0','4122-1.0','4122-2.0',
                    '4124-0.0','4124-1.0','4124-2.0']

temp=[]
for colname in df_truth.columns:
    mape=0
    amputated=np.array(df_amputated[colname])
    location=np.argwhere(np.isnan(amputated))
    n=len(location)
    truth=np.array(df_truth[colname])
    imputed=np.array(df_imputed[colname])
    for i in location:
       mape+=abs((truth[i]-imputed[i])/(truth[i]))
    mape=mape/n
    temp.append(mape[0]*100)
            
print (temp)

method=pd.DataFrame(temp,columns=(['Random indicator for non-ignorable data']))
merged=pd.concat([df_metrics,method],axis=1)
merged.to_csv('imputation_accuracy.csv')

