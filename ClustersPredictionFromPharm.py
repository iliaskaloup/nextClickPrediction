# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 22:51:36 2019

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

seed = 123
np.random.seed(seed)

milli_sec0 = int(round(time.time() * 1000))

training_data=pd.read_csv('dataSessions.csv', header=None, sep =',')
#training_data=training_data.iloc[0:397077,:] gia validation

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

#shuffle............
training_data=np.asarray(training_data)
indices = np.arange(training_data.shape[0])
np.random.shuffle(indices)
training_data = training_data[indices]
training_data=pd.DataFrame(training_data)

lenght=np.asarray(lenght)
lenght = lenght[indices]

timesteps=training_data.shape[1]

itemId2=training_data.values
itemId2=itemId2.flatten() 
features=itemId2.flatten()

trainSize=timesteps*(int(0.80*len(training_data)))
trainFeatures=features[0:trainSize]
testFeatures=features[trainSize:]

itemId2=pd.DataFrame(itemId2)
itemId2=itemId2[0].unique()
zeroIndex = np.where(itemId2 == 0)
itemId2=np.delete(itemId2, (zeroIndex), axis=0)
itemId2=pd.DataFrame(itemId2)

labelencoder = LabelEncoder()
itemId2['Labeled']=itemId2[0]
itemId2.iloc[:,1]=labelencoder.fit_transform(itemId2.iloc[:,0])

#trainSize=18048
trainSize=(int(0.80*len(training_data)))
train_lenght=lenght[0:trainSize]
test_lenght=lenght[trainSize:]
testSize=len(test_lenght)
training_data=np.asarray(training_data)
training_data2=training_data[0:trainSize]
testing_data=training_data[trainSize:]
dataFrameTrain=pd.DataFrame(training_data2)
testing_data=pd.DataFrame(testing_data)

def generator(X_data, batch_size,lenght,itemId2):

    samples_per_epoch = X_data.shape[0]
    number_of_batches = int(samples_per_epoch/batch_size)
    counter=0
    
    s=(batch_size,timesteps-1,len(itemId2))
    #ss=(batch_size,timesteps-1)
      
 
    while 1:
        
        X=np.zeros(s)
        X=X.astype(int)
        y=np.zeros(s)
        y=y.astype(int)
        #test_features=np.zeros(ss)
        #labels=np.zeros(ss)
        
        for i in range (0,batch_size):
            #print("i=%d"%i)
            #print("metrhths=%d"%counter)
            for j in range (0,lenght[i+counter*batch_size]-1):   
                current=X_data.iloc[i+counter*batch_size,j]
                index=int(itemId2[itemId2[0]==current].index[0])
                X[i,j,itemId2.Labeled[index]]=1
                #test_features[i,j]=itemId2.Labeled[index]
                nextItem=X_data.iloc[i+counter*batch_size,j+1]
                index=int(itemId2[itemId2[0]==nextItem].index[0])
                y[i,j,itemId2.Labeled[index]]=1
                #labels[i,j]=itemId2.Labeled[index]

        counter += 1
        yield X,y

        #restart counter to yeild data in the next epoch as well
        if counter >= (number_of_batches):
            counter = 0

def test_generator2(X_data, batch_size, lenght, itemId2):

    samples_per_epoch = X_data.shape[0]
    number_of_batches = int(samples_per_epoch/batch_size)
    counter=0
    
    s=(batch_size,timesteps-1,len(itemId2))
    #ss=(batch_size,timesteps-1)  
 
    while 1:
        
        X=np.zeros(s)
        X=X.astype(int)
        y=np.zeros(s)
        y=y.astype(int)
        #test_features=np.zeros(ss)
        #labels=np.zeros(ss)
        
        for i in range (0,batch_size):
            #print("i=%d"%i)
            #print("metrhths=%d"%counter)
            for j in range (0,lenght[i+counter*batch_size]-1):   
                current=X_data.iloc[i+counter*batch_size,j]
                index=int(itemId2[itemId2[0]==current].index[0])
                X[i,j,itemId2.Labeled[index]]=1
                #test_features[i,j]=itemId2.Labeled[index]
                nextItem=X_data.iloc[i+counter*batch_size,j+1]
                index=int(itemId2[itemId2[0]==nextItem].index[0])
                y[i,j,itemId2.Labeled[index]]=1
                #labels[i,j]=itemId2.Labeled[index]

        counter += 1
        yield X,y

        #restart counter to yeild data in the next epoch as well
        if counter >= (number_of_batches):
            counter = 0

#network
#tanh/sigmoid  scale[0,1]/[-1,1]
#class_weight=class_weight.compute_class_weight('balanced',np.unique(labels),labels)
BS=64
iterator2=test_generator2(testing_data,BS,test_lenght,itemId2)
my_init = keras.initializers.glorot_uniform(seed=seed)
model=Sequential()
model = load_model('PharmItemsNadam.h5')
        
milli_sec2 = int(round(time.time() * 1000))

#evaluation
def test_generator(X_data, batch_size, lenght, itemId2):

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
        
        for i in range (0,batch_size):
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

        counter += 1
        yield X,y,test_features,labels

        #restart counter to yeild data in the next epoch as well
        if counter >= (number_of_batches):
            counter = 0
            
print("Evaluating...")
milli_sec3 = int(round(time.time() * 1000))

number_of_elements = 1
recommended=[]
count=0
count2=0
iterator=test_generator(testing_data,BS,test_lenght,itemId2)
items_clusters = pd.read_csv('products2.csv', sep =',')

for k in range(0,len(testing_data),BS):
    #print("k=%d"%k)
    X_test,y_test,test_features,test_labels=next(iterator)
    
    preds = model.predict_classes(X_test, verbose=0)
    predictions = model.predict_proba(X_test, verbose=0) 
      
    for i in range (0,predictions.shape[0]):
        if (i+k<len(testing_data)):
            for j in range (0,test_lenght[i+k]-1):
                index3 = itemId2[ itemId2.iloc[:,1] == test_labels[i,j]]
                testClusters = items_clusters[ items_clusters["ProductId2"]==index3.iloc[0,0]]
                testCluster = testClusters["Cluster"].unique()
                
                temp=predictions[i,j,:]
                #sort=heapq.nlargest(number_of_elements, temp)
                idx = (-temp).argsort()[:number_of_elements]
                #temp=pd.DataFrame(temp)
                for z in range(0,number_of_elements):
                    #index=int(temp[temp[0]==sort[z]].index[0])
                    index=idx[z]
                    recommended.append(index)
                    if (test_labels[i,j] == index):
                        count=count+1
                        count2=count2+1
                        break
                    else:
                        index2=itemId2[ itemId2.iloc[:,1] == index]
                        table2=items_clusters[ items_clusters["ProductId2"]==index2.iloc[0,0]] 
                        table3=table2["Cluster"].unique()
                        if (table3 == testCluster):
                            count2=count2+1
                            break

count2=float(count2)  
recommended=pd.DataFrame(recommended)
#v=recommended[0].value_counts()
r=recommended[0].unique() 

denom=0
for i in range(0,len(test_lenght)):
    denom = denom + test_lenght[i] - 1
    
recall20 = count2/denom
recall20 = recall20*100
recall20 = round(recall20,2)
milli_sec4 = int(round(time.time() * 1000))
print("whole programme's time %d"%(milli_sec4-milli_sec0))
print("evaluation time %d"%(milli_sec4-milli_sec3))
print("recall@20: %.2f%%" % recall20)
print("r= %.2f%%" % len(r))
count=float(count)  
recall20 = count/denom
recall20 = recall20*100
recall20 = round(recall20,2)
print("count: %.2f%%" % recall20)

