#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Nov  8 23:18:36 2022

@author: daniel
"""

import pandas as pd
import numpy as np
from scipy.stats import zscore

df=pd.read_csv("ground_truth(3tp).csv",
               usecols=['21022-0.0','4100-0.0','4100-1.0','4100-2.0',
                        '4101-0.0','4101-1.0','4101-2.0',
                        '4103-0.0','4103-1.0','4103-2.0',
                        '4105-0.0','4105-1.0','4105-2.0',
                        '4119-0.0','4119-1.0','4119-2.0',
                        '4120-0.0','4120-1.0','4120-2.0',
                        '4122-0.0','4122-1.0','4122-2.0',
                        '4124-0.0','4124-1.0','4124-2.0'])

colnames=df.columns

#dropping random values
for i in range (0,len(colnames)-1,3):
    df[colnames[i]]=df[colnames[i]].sample(frac=0.83)

for i in range (1,len(colnames)-1,3):
    df[colnames[i]]=df[colnames[i]].sample(frac=0.1)

for i in range (2,len(colnames)-1,3):
    df[colnames[i]]=df[colnames[i]].sample(frac=0.17)
df.to_csv('3tp_amputated.csv')

