# -*- coding: utf-8 -*-
"""
Created on Sat Sep 21 15:51:02 2019

@author: Ilias
"""

import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from pandas import read_csv
from datetime import datetime

seed = 123
np.random.seed(seed)
clicksTrain=pd.read_csv('yoochoose-clicks.dat', header=None, sep =',')
#filter out less than x=100
items=clicksTrain.iloc[:,2].value_counts()
items=items[items > 100]
items=items.index.values
clicksTrain = clicksTrain[clicksTrain.iloc[:,2].isin(items)]

itemIdTrain=clicksTrain.iloc[:,2].unique()

clicksTest=pd.read_csv('yoochoose-test.dat', header=None, sep =',')
clicksTest = clicksTest[clicksTest.iloc[:,2].isin(itemIdTrain)]

clicks0 = pd.concat([clicksTrain, clicksTest])
#clicks0=clicks0.iloc[0:22716,:]
clicks=clicks0.iloc[:,[0,2]]
sessions=clicks0.iloc[:,0]
#mhkos kathe session
sessionLenght=clicks[0].value_counts()
sessionLenght=pd.DataFrame(sessionLenght)
sessionLenght=sessionLenght.rename(columns={0: "Times"})
sessionLenght["SessionId"]=sessionLenght.index.values
sessionLenght=sessionLenght.sort_values(by=["SessionId"])

maxLenght=max(sessionLenght["Times"])
meanLenght=round(sessionLenght.loc[:,"Times"].mean())

clicks=clicks[2]
clicks=pd.DataFrame(clicks)
labelencoder1 = LabelEncoder()
clicks["Ids"]=labelencoder1.fit_transform(clicks[2])

#classes
itemId2=clicks['Ids'].unique()
clicks['Ids'] = clicks['Ids'].replace(0, max(itemId2)+1)
itemId2=clicks['Ids'].unique()
print("Number of items=%d"%len(itemId2))

timesteps=35
sessionsNum=len(sessions.unique())
data=[]
k=0
items=clicks.iloc[:,1]
items=np.asarray(items)

for i in range(0,sessionsNum):
    lenght=sessionLenght.iloc[i,0]
    for j in range(0,lenght):
        data.append(items[k])
        k=k+1
        #print("k=%d"%k)
    if (lenght<timesteps):
        for p in range(0,(timesteps-lenght)):
            data.append(0)
    elif (lenght>timesteps):
        div=lenght//timesteps
        lenght2=lenght-(div*timesteps)
        for z in range(0,(timesteps-lenght2)):
            data.append(0)

data=np.asarray(data)
shape1=int(data.shape[0]/timesteps)
shape2=timesteps
data=np.reshape(data,(shape1,shape2))
data = pd.DataFrame(data)
#keep only sessions me lenght >1 
data=data[ data.iloc[:,1] != 0 ]
data.to_csv ('RecSysFullData.csv', index = None, header=False)







