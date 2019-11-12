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
import string 
import heapq
from keras.layers import Masking
from keras.callbacks import CSVLogger
from keras.callbacks import EarlyStopping, ModelCheckpoint

seed = 123
np.random.seed(seed)

training_data=pd.read_csv('C:/Users/Ilias/Desktop/pharm/dataSessions.csv', header=None, sep =',')
training_data=training_data.iloc[0:22716,:]

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
np.savetxt("C:/Users/Ilias/Desktop/featuresPharm.txt", (trainFeatures), fmt="%d")
np.savetxt("C:/Users/Ilias/Desktop/testFeaturesPharm.txt", (testFeatures), fmt="%d")

itemId2=pd.DataFrame(itemId2)
itemId2=itemId2[0].unique()
zeroIndex = np.where(itemId2 == 0)
itemId2=np.delete(itemId2, (zeroIndex), axis=0)
np.savetxt("C:/Users/Ilias/Desktop/itemsPharm.txt", (itemId2), fmt="%d")
itemId2=pd.DataFrame(itemId2)

labelencoder = LabelEncoder()
itemId2['Labeled']=itemId2[0]
itemId2.iloc[:,1]=labelencoder.fit_transform(itemId2.iloc[:,0])

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
    sy=(batch_size,len(itemId2))
 
    while 1:
        X=np.zeros(s)
        X=X.astype(int)
        y=np.zeros(sy)
        y=y.astype(int)
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
            y[i,itemId2.Labeled[index]]=1
                #labels[i,j]=itemId2.Labeled[index]

        counter += 1
        #flatten data for mlp
        X = np.reshape(X,(X.shape[0],X.shape[1]*X.shape[2]))
        yield X,y

        #restart counter to yeild data in the next epoch as well
        if counter >= (number_of_batches):
            counter = 0

def test_generator2(X_data, batch_size, lenght, itemId2):

    samples_per_epoch = X_data.shape[0]
    number_of_batches = int(samples_per_epoch/batch_size)
    counter=0
    
    s=(batch_size,timesteps-1,len(itemId2))
    sy=(batch_size,len(itemId2))
 
    while 1:
        X=np.zeros(s)
        X=X.astype(int)
        y=np.zeros(sy)
        y=y.astype(int)
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
            y[i,itemId2.Labeled[index]]=1
                #labels[i,j]=itemId2.Labeled[index]

        counter += 1
        #flatten data for mlp
        X = np.reshape(X,(X.shape[0],X.shape[1]*X.shape[2]))
        yield X,y

        #restart counter to yeild data in the next epoch as well
        if counter >= (number_of_batches):
            counter = 0


#network
#tanh/sigmoid  scale[0,1]/[-1,1]
#class_weight=class_weight.compute_class_weight('balanced',np.unique(labels),labels)
nb_epoch=5
BS=128
iterator2=test_generator2(testing_data,BS,test_lenght,itemId2)
my_init = keras.initializers.glorot_uniform(seed=seed)
model=Sequential()
model.add(Dense(units=500, kernel_initializer=my_init, input_dim=(timesteps-1)*len(itemId2)))  
model.add(Activation('tanh')) 
#model.add(Dropout(0.2))
    
model.add(Dense(units=500, kernel_initializer=my_init))  
model.add(Activation('tanh')) 

model.add(Dense(units = 500, kernel_initializer=my_init))
model.add(Activation('tanh')) 

model.add(Dense(units = len(itemId2), kernel_initializer=my_init))
#model.add(Dense(units = len(itemId2), kernel_initializer=my_init))
model.add(Activation('softmax'))#Output
#model.add(Activation('tanh'))#Output
#activation function 
#sgd = optimizers.Adagrad(lr=0.001, epsilon=None, decay=1e-6)
#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.98, nesterov=True )
sgd = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)
#sgd = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False )
#sgd = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy')  
#model.compile(optimizer = sgd, loss = 'binary_crossentropy')  
#fit the model
milli_sec = int(round(time.time() * 1000))
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
csv_logger = CSVLogger('C:/Users/Ilias/Desktop/log2.csv', append=True, separator=',')
mc = ModelCheckpoint('C:/Users/Ilias/Desktop/best_model.h5', monitor='val_rec', mode='max', verbose=1, save_best_only=True)
history = model.fit_generator(generator(dataFrameTrain,BS,train_lenght,itemId2),validation_data=next(iterator2),epochs=nb_epoch, callbacks=[csv_logger,es,mc],steps_per_epoch = int(trainSize//BS),validation_steps=int(testSize // BS),use_multiprocessing=False)
#model.fit(train_data, train_labels, epochs = nb_epoch, batch_size = BS, shuffle=False, verbose=1)
'''
csv_logger = CSVLogger('log.csv', append=True, separator=';')
model.fit(X_train, Y_train, callbacks=[csv_logger])
'''
milli_sec2 = int(round(time.time() * 1000))
print(milli_sec2-milli_sec)

#plot training history
plt.plot(history.history['loss'], label='train')
pic=plt.plot(history.history['val_loss'], label='test')
plt.legend()
#plt.show()
plt.savefig('C:/Users/Ilias/Desktop/train-test_loss_pharm.jpg')


path='C:/Users/Ilias/Desktop/my_modelPharm22716.h5'
model.save(path)
#from keras.models import load_model
#model = load_model('my_modelPharm22716.h5')

#evaluation
def test_generator(X_data, batch_size, lenght, itemId2):

    samples_per_epoch = X_data.shape[0]
    number_of_batches = int(samples_per_epoch/batch_size)
    counter=0
    
    s=(batch_size,timesteps-1,len(itemId2))
    ss=(batch_size,timesteps-1)
    sy=(batch_size,len(itemId2))
    sss=(batch_size)
 
    while 1:
        
        X=np.zeros(s)
        X=X.astype(int)
        y=np.zeros(sy)
        y=y.astype(int)
        test_features=np.zeros(ss)
        test_features=test_features.astype(int)
        labels=np.zeros(sss)
        labels=labels.astype(int)
        
        for i in range (0,batch_size):
            #print("i=%d"%i)
            #print("metrhths=%d"%counter)
            for j in range (0,lenght[i+counter*batch_size]-1):   
                current=X_data.iloc[i+counter*batch_size,j]
                index=int(itemId2[itemId2[0]==current].index[0])
                X[i,j,itemId2.Labeled[index]]=1
                nextItem=X_data.iloc[i+counter*batch_size,j+1]
                index=int(itemId2[itemId2[0]==nextItem].index[0])
            test_features[i,j]=itemId2.Labeled[index]
            y[i,itemId2.Labeled[index]]=1
            labels[i]=itemId2.Labeled[index]

        counter += 1
        #flatten data for mlp
        X = np.reshape(X,(X.shape[0],X.shape[1]*X.shape[2]))
        yield X,y,test_features,labels

        #restart counter to yeild data in the next epoch as well
        if counter >= (number_of_batches):
            counter = 0

print("Evaluating...")
number_of_elements = 20
count=0
recommended=[]
iterator=test_generator(testing_data,BS,test_lenght,itemId2)

for k in range(0,len(testing_data),BS):
    #print("k=%d"%k)
    X_test,y_test,test_features,test_labels=next(iterator)
    
    preds = model.predict_classes(X_test, verbose=0)
    predictions = model.predict_proba(X_test, verbose=0) 
      
    for i in range (0,predictions.shape[0]):
        if (i+k<len(testing_data)):
            temp=predictions[i,:]
            #sort=heapq.nlargest(number_of_elements, temp)
            idx = (-temp).argsort()[:number_of_elements]
            #temp=pd.DataFrame(temp)
            for z in range(0,number_of_elements):
                #index=int(temp[temp[0]==sort[z]].index[0])
                index=idx[z]
                recommended.append(index)
                if (test_labels[i] == index):
                    count=count+1
                        
count=float(count)
recommended=pd.DataFrame(recommended)
recommended[0].value_counts()
r=recommended[0].unique() 

denom=len(test_lenght)
    
recall20 = count/denom
recall20 = recall20*100
recall20 = round(recall20,2)
print("recall@20: %.2f%%" % recall20)
print("r= %.2f%%" % len(r))
#diagrammata
#log = read_csv('C:/Users/Ilias/Desktop/log.csv')
#pic=log.plot(x ='epoch', y='loss', kind = 'scatter').get_figure()
#pic.savefig('C:/Users/Ilias/Desktop/epochs-loss.jpg')





