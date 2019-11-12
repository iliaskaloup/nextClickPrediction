# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 03:30:35 2019

@author: Ilias
"""

# -*- coding: utf-8 -*-
"""
Created on Sun Apr  7 17:53:01 2019

@author: Ilias
"""
import tensorflow as tf
import keras as keras
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers import Flatten
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
from keras.regularizers import l2
from keras.constraints import max_norm

seed = 123*2
np.random.seed(seed)

# load dataset
df = read_csv('C:/Users/Ilias/Desktop/dataset1998/PRdataDataFrame.csv')
df=df.drop('Index',axis=1)
'''
df = df[ (df.item != 1008) & (df.item != 1034) & (df.item != 1004) & (df.item != 1018) & (df.item != 1001) & (df.item != 1026) 
& (df.item != 1003) & (df.item != 1009) & (df.item != 1017)  & (df.item != 1035) & (df.item != 1025) & (df.item != 1041) 
& (df.item != 1037) & (df.item != 1032) & (df.item != 1038) & (df.item != 1030) & (df.item != 1020)  ]
'''
#df['item'].value_counts()

#create sessions
sessions=df.sessions.unique()
data=[]
for i in range (0,len(sessions)):
    d=df[ df.sessions == sessions[i] ]
    x=[]
    x=d.item
    x=x.tolist()
    if (len(d) >1 ):
        data.append(x)

training_data = pd.DataFrame(data)
training_data = training_data.fillna(0)
#training_data=shuffle(training_data, random_state=seed)

timesteps=training_data.shape[1]
#timesteps=2

itemId2=training_data.values
itemId2=itemId2.flatten() 
features=itemId2.flatten()

trainSize=timesteps*(int(0.80*len(training_data)))
trainFeatures=features[0:trainSize]
testFeatures=features[trainSize:]
np.savetxt("C:/Users/Ilias/Desktop/features.txt", (trainFeatures), fmt="%d")
np.savetxt("C:/Users/Ilias/Desktop/testFeatrues.txt", (testFeatures), fmt="%d")

itemId2=pd.DataFrame(itemId2)
itemId2=itemId2[0].unique()
zeroIndex = np.where(itemId2 == 0)
itemId2=np.delete(itemId2, (zeroIndex), axis=0)
np.savetxt("C:/Users/Ilias/Desktop/items.txt", (itemId2), fmt="%d")
itemId2=pd.DataFrame(itemId2)

labelencoder = LabelEncoder()
itemId2['Labeled']=itemId2[0]
itemId2.iloc[:,1]=labelencoder.fit_transform(itemId2.iloc[:,0])

trainSize2=(int(0.80*len(data)))
training_data2=data[0:trainSize2]
train_lenght=[]
for i in range (0,len(training_data2)):
    train_lenght.append(len(training_data2[i]))

testing_data=data[trainSize2:]
test_lenght=[]
for i in range (0,len(testing_data)):
    test_lenght.append(len(testing_data[i]))
    
lenght=train_lenght+test_lenght

s=(len(training_data),timesteps-1,len(itemId2))
ss=(len(training_data),timesteps-1)
sy=(len(training_data),len(itemId2))
sss=(len(training_data))
X=np.zeros(s)
X=X.astype(int)
y=np.zeros(sy)
y=y.astype(int)
test_features=np.zeros(ss)
test_features=test_features.astype(int)
labels=np.zeros(sss)
labels=labels.astype(int)

for i in range (0,len(training_data)):
    for j in range (0,lenght[i]-1):                                                                                                                 
        current=training_data.iloc[i,j]
        index=int(itemId2[itemId2[0]==current].index[0])
        X[i,j,itemId2.Labeled[index]]=1
        test_features[i,j]=itemId2.Labeled[index]
        nextItem=training_data.iloc[i,j+1]
        index=int(itemId2[itemId2[0]==nextItem].index[0])
    y[i,itemId2.Labeled[index]]=1
    labels[i]=itemId2.Labeled[index]
        
trainSize=int(0.80*len(training_data))
testSize=len(training_data)-trainSize
train_data=X[0:trainSize,:,:]
train_labels=y[0:trainSize,:]
test_data=X[trainSize:len(training_data),:,:]
test_labels=y[trainSize:len(training_data),:]
test_data2=test_features[trainSize:len(training_data),:]
test_labels2=labels[trainSize:len(training_data)]

#network
#tanh/sigmoid  scale[0,1]/[-1,1]
#class_weight=class_weight.compute_class_weight('balanced',np.unique(labels),labels)
nb_epoch=5
BS=64
my_init = keras.initializers.glorot_uniform(seed=seed)
model=Sequential()
model.add(Masking(mask_value=0, input_shape=(train_data.shape[1], train_data.shape[2])))
model.add(GRU(units=200, kernel_initializer=my_init, stateful=False))  
model.add(Activation('tanh')) 
#model.add(Dropout(0.2))
    
'''
model.add(LSTM(units=200, kernel_initializer=my_init, return_sequences=True))  
model.add(Activation('tanh'))
'''
model.add(Dense(units = len(itemId2), kernel_initializer=my_init))
#model.add(Dense(units = len(itemId2), kernel_initializer=my_init))
model.add(Activation('softmax'))#Output
#model.add(Activation('tanh'))#Output
#activation function 
#sgd = optimizers.Adagrad(lr=0.001, epsilon=None, decay=1e-6)
#sgd = optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.98, nesterov=True )
#sgd = optimizers.RMSprop(lr=0.001, rho=0.9, epsilon=1e-8, decay=0.0)
#sgd = optimizers.adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=1e-6, amsgrad=False )
sgd = optimizers.Nadam(lr=0.002, beta_1=0.9, beta_2=0.999, epsilon=None, schedule_decay=0.004)
model.compile(optimizer = sgd, loss = 'categorical_crossentropy')  
#model.compile(optimizer = sgd, loss = 'binary_crossentropy')  
es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=2)
#fit the model
milli_sec = int(round(time.time() * 1000))
csv_logger = CSVLogger('C:/Users/Ilias/Desktop/log.csv', append=True, separator=',')
#model.fit(train_data, train_labels, epochs = nb_epoch, batch_size = BS, shuffle=False, verbose=1)

mc = ModelCheckpoint('C:/Users/Ilias/Desktop/best_model.h5', monitor='val_rec', mode='max', verbose=1, save_best_only=True)
history = model.fit(train_data, train_labels, validation_data=(test_data, test_labels), epochs=nb_epoch, batch_size=BS, shuffle=False, verbose=1, callbacks=[csv_logger,es,mc])

milli_sec2 = int(round(time.time() * 1000))
print(milli_sec2-milli_sec)

#plot training history
plt.plot(history.history['loss'], label='train')
pic=plt.plot(history.history['val_loss'], label='test')
plt.legend()
#plt.show()
plt.savefig('C:/Users/Ilias/Desktop/train-test_loss.jpg')

#evaluation
preds = model.predict_classes(test_data, verbose=0)
predictions = model.predict_proba(test_data, verbose=0) 
      
#training_data[0].value_counts()

pred=preds.flatten()
pred=pd.DataFrame(pred)
pred=pred[0].unique()
len(pred)

path='C:/Users/Ilias/Desktop/my_model.h5'
model.save(path)
#model = load_model('my_model.h5')

print("Evaluating...")
#evaluation
number_of_elements = 20
count=0
recommended=[]
for i in range (0,predictions.shape[0]):
    temp=predictions[i,:]
    #sort=heapq.nlargest(number_of_elements, temp)
    idx = (-temp).argsort()[:number_of_elements]
    #temp=pd.DataFrame(temp)
    for z in range(0,number_of_elements):
        #index=int(temp[temp[0]==sort[z]].index[0])
        index=idx[z]
        recommended.append(index)
        if (test_labels2[i] == index):
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





