# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 23:54:49 2019

@author: Ilias
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 19:30:57 2019

@author: Ilias
"""

import tensorflow as tf
import keras as keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from keras import optimizers
from keras.regularizers import l2
from sklearn import metrics
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import scale
from sklearn.utils import shuffle
import time
from keras.layers import LSTM
from keras.layers import GRU
from pandas import read_csv
from keras.layers import TimeDistributed
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import class_weight
from keras.layers import Masking
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import load_model
import random

seed = 123
np.random.seed(seed)

training_data=pd.read_csv('dataSessions_clusters.csv', header=None, sep =',')
#training_data=training_data.iloc[0:22700,:] 

items=pd.read_csv('dataSessions.csv', header=None, sep =',')
#items=items.iloc[0:22700,:] 
itemId1=items.values
itemId1=itemId1.flatten() 
itemId1=pd.DataFrame(itemId1)
itemId1=itemId1[0].value_counts()
itemId1=pd.DataFrame(itemId1)
itemId1=itemId1.rename(columns={0: "Times"})
itemId1[0]=itemId1.index.values
itemId1=pd.DataFrame(itemId1)

items_clusters = pd.read_csv('products2.csv', sep =',')
items_clusters.Cluster = items_clusters.Cluster.replace(0, max(items_clusters["Cluster"])+1)

#euresh new session_lenght
lenght=[]
for i in range(0,len(training_data)):
    count=0
    for j in range(0,training_data.shape[1]):
        temp=training_data.iloc[i,j]
        if (temp==0):
            break
        else:
            count=count+1
    lenght.append(count)

#shuffle
'''
training_data=np.asarray(training_data)
indices = np.arange(training_data.shape[0])
np.random.shuffle(indices)
training_data = training_data[indices]
training_data=pd.DataFrame(training_data)

items=np.asarray(items)
items = items[indices]

lenght=np.asarray(lenght)
lenght = lenght[indices]
'''
training_data=pd.DataFrame(training_data)
items=np.asarray(items)
lenght=np.asarray(lenght)


timesteps=training_data.shape[1]

itemId2=training_data.values
itemId2=itemId2.flatten() 
features=itemId2.flatten()

itemId2=pd.DataFrame(itemId2)
itemId2=itemId2[0].unique()
zeroIndex = np.where(itemId2 == 0)
itemId2=np.delete(itemId2, (zeroIndex), axis=0)
itemId2=pd.DataFrame(itemId2)

items_clusters = items_clusters[items_clusters['Cluster'].isin(itemId2[0])]

labelencoder = LabelEncoder()
itemId2['Labeled']=itemId2[0]
itemId2.iloc[:,1]=labelencoder.fit_transform(itemId2.iloc[:,0])

trainSize=(int(0.80*len(training_data)))
train_lenght=lenght[0:trainSize]
test_lenght=lenght[trainSize:]
testSize=len(test_lenght)
training_data=np.asarray(training_data)
training_data2=training_data[0:trainSize]
trainItems=items[0:trainSize]
testing_data=training_data[trainSize:]
testItems=items[trainSize:]
dataFrameTrain=pd.DataFrame(training_data2)
testing_data=pd.DataFrame(testing_data)

BS=128
model=Sequential()
model = load_model('my_modelPharmFinalClusters.h5')
#items from clusters
'''
TABLE_SIZE = 1000000
denom = 0.
power=power = 0.75
table=[]
clusters=[]
unique = itemId1.iloc[:,1]
unique=np.asarray(unique)
for i in range(1,len(unique)):
    cluster=items_clusters[ items_clusters.ProductId2 == unique[i] ]
    cluster=np.asarray(cluster)
    clusters.append(cluster[0,4])

unique=pd.DataFrame(unique)  
unique["Clusters"]=clusters
unique=np.asarray(unique)    
counts = itemId1["Times"]
counts=np.asarray(counts)
for i in range(0,len(counts)):
    denom = denom + pow(counts[i], power)
for i in range(0,len(counts)):
    numerator = pow(counts[i], power)
    limit = int(numerator*TABLE_SIZE/denom)
    for j in range(0,limit):
        table.append(unique[i,:])
table=np.asarray(table)
#table=shuffle(table)
indices = np.arange(table.shape[0])
np.random.shuffle(indices)
table = table[indices]
table=pd.DataFrame(table)
'''
#evaluation
def test_generator(X_data, batch_size, lenght, itemId2, testItems):

    samples_per_epoch = X_data.shape[0]
    number_of_batches = int(samples_per_epoch/batch_size)
    counter=0
    
    s=(batch_size,timesteps-1,len(itemId2))
    ss=(batch_size,timesteps-1) 
    
    while 1:
        
        X=np.zeros(s)
        X=X.astype(int)
        y=np.zeros(s)
        y=y.astype(int)
        test_features=np.zeros(ss)
        labels=np.zeros(ss) 
        #items=np.zeros(ss)
        
        for i in range (0,batch_size):
            items[i,:]=testItems[i+counter*batch_size,:]
            #print("i=%d"%i)
            #print("metrhths=%d"%counter)
            for j in range (0,lenght[i+counter*batch_size]-1):   
                current=X_data.iloc[i+counter*batch_size,j]
                index=int(itemId2[itemId2[0]==current].index[0])
                X[i,j,itemId2.Labeled[index]]=1
                test_features[i,j]=itemId2.Labeled[index]
                nextItem=X_data.iloc[i+counter*batch_size,j+1]
                index=int(itemId2[itemId2[0]==nextItem].index[0])
                y[i,j,itemId2.Labeled[index]]=1
                labels[i,j]=itemId2.Labeled[index]
                #items[i,j]=testItems[i+counter*batch_size,j]

        counter += 1
        yield X,y,test_features,labels,items

        #restart counter to yeild data in the next epoch as well
        if counter >= (number_of_batches):
            counter = 0
            
print("Evaluating...")
milli_sec3 = int(round(time.time() * 1000))

number_of_elements = 20
count=0
count0=0
iterator=test_generator(testing_data,BS,test_lenght,itemId2,testItems)

for k in range(0,len(testing_data),BS):
    #print("k=%d"%k)
    X_test,y_test,test_features,test_labels,test_items=next(iterator)
    
    preds = model.predict_classes(X_test, verbose=0)
    predictions = model.predict_proba(X_test, verbose=0) 
      
    for i in range (0,predictions.shape[0]):
        if (i+k<len(testing_data)):
            for j in range (0,test_lenght[i+k]-1):
                temp=predictions[i,j,:]
                idx = (-temp).argsort()[:number_of_elements]
                for z in range(0,number_of_elements):
                    index=idx[z]
                    if (test_labels[i,j] == index):
                        count0=count0+1
                        index2=itemId2[ itemId2.iloc[:,1] == index]
                        #index2=index2.iloc[:,0]
                        #table2=table[ table.iloc[:,1]==index2]
                        table2=items_clusters[ items_clusters["Cluster"]==index2.iloc[0,0]] 
                        table3=table2["ProductId2"].tolist() 
                        if (len(table3)< 1):                                     
                            temp = random.sample(table3,k=len(table3))
                        else:
                            temp = random.sample(table3,k=1)
                        for cc in range(0,len(temp)):
                            if (temp[cc] == test_items[i,j]):
                                count=count+1
                        
count=float(count)  
denom=0
for i in range(0,len(test_lenght)):
    denom = denom + test_lenght[i] - 1    
recall20 = count/denom
recall20 = recall20*100
recall20 = round(recall20,2)
milli_sec4 = int(round(time.time() * 1000))
print("evaluation time %d"%(milli_sec4-milli_sec3))
print("recall@20: %.2f%%" % recall20)
count0=float(count0)
recall20=(count0/denom)*100
recall20 = round(recall20,2)
print("count0: %.2f%%" % recall20)
