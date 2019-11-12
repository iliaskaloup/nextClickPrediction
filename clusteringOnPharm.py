# -*- coding: utf-8 -*-
"""
Created on Mon Jun 24 19:57:17 2019

@author: Ilias
"""
import pandas as pd
import numpy as np
from kmodes.kprototypes import KPrototypes
from sklearn.cluster import KMeans
from kmodes.kmodes import KModes
from sklearn.preprocessing import LabelEncoder
import time
from sklearn.utils import shuffle

seed = 123
np.random.seed(seed)

df = pd.read_csv('C:/Users/Ilias/Desktop/pharm/products.csv', sep =';', names=["ProductId", "ManufactorerId", "Price", "Category"])
df = df.drop_duplicates(['ProductId'], keep='last')

#preprocessing
a=df.iloc[:,3]
a=pd.DataFrame(a)
df=df.iloc[:,0:3]
a.to_csv ('C:/Users/Ilias/Desktop/pharm/categories.csv', index = None, header=True)
a = pd.read_csv('C:/Users/Ilias/Desktop/pharm/categories.csv', sep =';')
a=a.iloc[:,0]
df.loc[:,'Category1']=pd.Series(a,index=df.index)
df=df.fillna("No Category")

labelencoder2 = LabelEncoder()
df.iloc[:,1]=labelencoder2.fit_transform(df.iloc[:,1])
labelencoder3 = LabelEncoder()
df.iloc[:,3]=labelencoder3.fit_transform(df.iloc[:,3])

#price categories
for i in range(len(df)):
    if (df.iloc[i,2]==0):
        df.iloc[i,2]="free"
    elif (df.iloc[i,2]<10):
        df.iloc[i,2]="bargain"
    elif (df.iloc[i,2]<30):
        df.iloc[i,2]="normal"
    elif (df.iloc[i,2]<30):
        df.iloc[i,2]="high normal"
    elif (df.iloc[i,2]<80):
        df.iloc[i,2]="high expensive"
    elif (df.iloc[i,2]<200):
        df.iloc[i,2]="expensive"
    elif (df.iloc[i,2]<1000):
        df.iloc[i,2]="very expensive"
    else:
        df.iloc[i,2]="huge"
        
df.to_csv ('C:/Users/Ilias/Desktop/pharm/products2.csv', index = None, header=True) #Don't forget to add '.csv' at the end of the path

#clustering
df = pd.read_csv('C:/Users/Ilias/Desktop/pharm/products2.csv', sep =',')
#df=df.drop(['Cluster'],axis=1)
X=df.drop(['ProductId'],axis=1)
#X=X.iloc[:,1]
#X=shuffle(X)

milli_sec = int(round(time.time() * 1000))
km = KModes(n_clusters=500, init='Huang', n_init=5, verbose=1)
clusters=km.fit_predict(X)
milli_sec2 = int(round(time.time() * 1000))
print(milli_sec2-milli_sec)
df.loc[:,'Cluster']=pd.Series(clusters,index=df.index)
clusters=pd.DataFrame(clusters)
print(clusters[0].value_counts())
df.to_csv ('C:/Users/Ilias/Desktop/pharm/products2.csv', index = None, header=True) 




    

