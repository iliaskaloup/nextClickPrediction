# -*- coding: utf-8 -*-
"""
Created on Sat Apr  6 21:27:19 2019

@author: Ilias
"""
import numpy as np
import pandas as pd
import statistics
import matplotlib.pyplot as plt

dataset = pd.read_csv('C:/Users/Ilias/Desktop/dataset1998/PRdata.txt')
itemId2=dataset.itemId[0:294] #classes-unique products 
data=dataset.iloc[294:,0:2] 
sessions=list(range(10001, 42712))
sessions=pd.DataFrame(sessions)
c=[]
for index, row in data.iterrows() :
    if (data.A[index]=='V'):
        c.append(data.itemId[index])
clicks=pd.DataFrame(c)#itemIds

#vriskw to mhkos tou kathe session
length=[]
count=0
for index, row in data.iterrows():
    if (data.A[index]=='V'):
        count=count+1
    elif (data.A[index]=='C'):
        length.append(count)
        count=0
length.append(count)
length=pd.DataFrame(length)
length.drop(length.index[0], inplace = True)

'''pd.value_counts(clicks[0]).plot.bar()
plt.title('click class histogram')
plt.xlabel('item')
plt.ylabel('Frequency')
clicks[0].value_counts()'''

'''pd.value_counts(length[0]).plot.bar()
plt.title('sessionLength frequency histogram')
plt.xlabel('item')
plt.ylabel('Frequency')
length[0].value_counts()'''

#create a general dataFrame of the dataset
l=[]
s=[]
it=[]
Idx=0
for i in range(0,len(length)):
    for j in range(0,int(length.iloc[i])):
        l.append(int(length.iloc[i]))
        s.append(int(sessions.iloc[i]))
        it.append(int(clicks.iloc[Idx+j]))
    Idx+=int(length.iloc[i])

d={'sessions':s,'length':l,'item':it}
df = pd.DataFrame(d)
#filter out sessions with only one click
#me 2 oi pinakes me unique
df=df[df.length != 1]
#df.to_csv('C:/Users/Ilias/Desktop/dataset1998/PRdataDataFrame.csv',index=False)
sessions2=df['sessions'].unique()
length2=length[length[0] != 1]#length2.mean()=4
'''pd.value_counts(length2[0]).plot.bar()
plt.title('sessionLength2 frequency histogram')
plt.xlabel('item')
plt.ylabel('Frequency')
length2[0].value_counts()'''






