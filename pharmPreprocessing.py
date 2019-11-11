# -*- coding: utf-8 -*-
"""
Created on Sun Jul 28 16:32:14 2019

@author: Ilias
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from pandas import read_csv

seed = 123
np.random.seed(seed)

#αναγωνση δεδομενων,αφαίρεση διπλότυπων 
clicks = pd.read_csv('C:/Users/Ilias/Desktop/pharm/clicks2.csv', sep =';', names=["ProductId", "CustomerId", "Timestamp", "SessionId"])
clicks = clicks.drop_duplicates( subset=None, keep='first')
#labelEncoding ProductId
items = pd.read_csv('C:/Users/Ilias/Desktop/pharm/products2.csv', sep =',')
items = items[items['ProductId'].isin(clicks["ProductId"])]
clicks = clicks[clicks['ProductId'].isin(items["ProductId"])]
labelencoder1 = LabelEncoder()
items["ProductId2"]=labelencoder1.fit_transform(items.iloc[:,0])
clicks["ProductId2"]=labelencoder1.fit_transform(clicks.iloc[:,0])
items.to_csv ('C:/Users/Ilias/Desktop/pharm/products2.csv', index = None, header=True) 
#ελεγχος συχνότητας προιόντων
freq=clicks["ProductId2"].value_counts()
freq=pd.DataFrame(freq)
freq=freq.rename(columns={"ProductId2": "Times"})
freq["ProductId2"]=freq.index.values
           
#antistoixish products me clusters
freq=freq.sort_values(by=["ProductId2"])
items = pd.read_csv('C:/Users/Ilias/Desktop/pharm/products2.csv', sep =',')
items=items.sort_values(by=["ProductId2"])
clicks=clicks.sort_values(by=["ProductId2"])
clusters=[]
for i in range(0,len(items)):
    temp=items.iloc[i,4]
    for j in range(0,freq.iloc[i,0]):
        clusters.append(temp)
clicks["Cluster"]=clusters
clicks.to_csv ('C:/Users/Ilias/Desktop/pharm/clicks3.csv', index = None, header=True) 

#ταξινόμηση με χρονολογική σειρά 
clicks = pd.read_csv('C:/Users/Ilias/Desktop/pharm/clicks3.csv', sep =',')
labelencoder = LabelEncoder()
clicks["SessionId"]=labelencoder.fit_transform(clicks.iloc[:,3])
clicks=clicks.sort_values(by=['Timestamp'])
clicks.to_csv ('C:/Users/Ilias/Desktop/pharm/clicks4.csv', index = None, header=True) 
clusterFrequency=clicks["Cluster"].value_counts()

#ταξινόμηση των κλικς ανα session 
clicks = pd.read_csv('C:/Users/Ilias/Desktop/pharm/clicks4.csv', sep =',')
sessions=clicks['SessionId'].unique() #αριθμός sessions->len(sessions)
sessions=pd.DataFrame(sessions)
sessions=sessions.sort_values(by=[0])
sessionsNum=len(sessions)

#mhkos kathe session
sessionLenght=clicks['SessionId'].value_counts()
sessionLenght=pd.DataFrame(sessionLenght)
sessionLenght=sessionLenght.rename(columns={"SessionId": "Times"})
sessionLenght["SessionId"]=sessionLenght.index.values
sessionLenght=sessionLenght.sort_values(by=["SessionId"])

ses=sessionLenght[sessionLenght.Times == 1]#sessions me mono ena click
clicks = clicks[~clicks['SessionId'].isin(ses["SessionId"])]#drop sessions me mono ena click
sessions = sessions[~sessions[0].isin(ses["SessionId"])]#drop sessions me mono ena click
sessionsNum=len(sessions)
sessionLenght = sessionLenght[~sessionLenght['SessionId'].isin(ses["SessionId"])]#drop sessions me mono ena click
maxLenght=max(sessionLenght["Times"])
meanLenght=round(sessionLenght.loc[:,"Times"].mean())

df=clicks[ clicks.SessionId == sessions.iloc[0,0] ]
for i in range(1,sessionsNum):
    print("i=%d"%i)
    d=clicks[ clicks.SessionId == sessions.iloc[i,0] ]
    df=pd.concat([df, d])
    
df.to_csv ('C:/Users/Ilias/Desktop/pharm/clicks5.csv', index = None, header=True) 

timesteps=35
#δημιουργία sessions από κλικς με σωσ΄τή σειρά -> το dataset μας
clicks = pd.read_csv('C:/Users/Ilias/Desktop/pharm/clicks5.csv', sep =',')
itemId2=clicks.Cluster.unique()
clicks.Cluster = clicks.Cluster.replace(0, max(itemId2)+1)
itemId2=clicks.Cluster.unique()
data=[]
k=0
clusters=clicks.iloc[:,5]
clusters=np.asarray(clusters)

sessionsNum=len(clicks['SessionId'].unique())
sesLen=[]
for i in range(0,sessionsNum):
    lenght=sessionLenght.iloc[i,0]
    for j in range(0,lenght):
        data.append(clusters[k])
        k=k+1
        print("k=%d"%k)
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
training_data = pd.DataFrame(data)
training_data.to_csv ('C:/Users/Ilias/Desktop/pharm/clicks6.csv', index = None, header=False) 
#keep only sessions me lenght >1 
training_data=training_data[ training_data.iloc[:,1] != 0 ]
training_data.to_csv ('C:/Users/Ilias/Desktop/pharm/dataSessions.csv', index = None, header=False)


