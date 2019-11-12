# -*- coding: utf-8 -*-
"""
Created on Fri Jul 26 01:06:09 2019

@author: Ilias
"""

import pandas as pd
import numpy as np
from pandas import read_csv
import random

seed = 123*2
random.seed(seed)

itemIds=itemId2['Labeled'].values
number_of_elements=20
count=0

for i in range(0,test_data2.shape[0]):
    for j in range(0,test_lenght[i]-1):
        temp = random.choices(itemIds,k=number_of_elements)
        for z in range(0,number_of_elements):
            if (test_labels2[i,j] == temp[z]):
                count=count+1
        
denom=0
for i in range(0,len(test_lenght)):
    denom = denom + test_lenght[i] - 1
    
recall20 = count/denom
recall20 = recall20*100
recall20 = round(recall20,2)
print("recall@20: %.2f%%" % recall20)



