# -*- coding: utf-8 -*-
"""
Created on Wed Dec  2 10:37:32 2020

@author: Happy Family
"""

import numpy as np
from sko.GA import GA
import pandas as pd
import matplotlib.pyplot as plt
import math
from math import sin,cos
from math import pi
import matplotlib.pyplot as plt
import pandas as pd
from pandas import read_csv
import time
import math
import tensorflow as tf

from keras.layers.core import Dense, Activation,Dropout
from keras.models import Sequential 
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

LOOK_BACK = 125

Train1 = read_csv('./dataset/train/train4pure.csv')
Test1 = read_csv('./dataset/test/train1compressed.csv')
train_cd = Train1['Cd'].values
train_theta = Train1['theta'].values
test_cd = Test1['Cd'].values
test_theta = Test1['theta'].values
train_cip = Train1['Cip'].values

train_cd = train_cd.reshape(-1,1)
test_cd = test_cd.reshape(-1,1)
train_theta = train_theta.reshape(-1,1)
test_theta = test_theta.reshape(-1,1)
train_cip = train_cip.reshape(-1,1)

scaler_cd = MinMaxScaler(feature_range=(-1,1))
scaler_cd.fit(train_cd)
train_cd_scaled = scaler_cd.transform(train_cd)
test_cd_scaled = scaler_cd.transform(test_cd)

scaler_theta = MinMaxScaler(feature_range=(-1,1))
scaler_theta.fit(train_theta)
train_theta_scaled = scaler_theta.transform(train_theta)
test_theta_scaled = scaler_theta.transform(test_theta)  

scaler_cip = MinMaxScaler(feature_range=(-1,1))
scaler_cip.fit(train_cip)
train_cip_scaled = scaler_cip.transform(train_cip)

def buildDataset(cddata, look_back=1):
    dataX, dataY = [], []
    for i in range(len(cddata) - look_back - 1):
        a = cddata[i:(i+look_back), 0]
        dataX.append(a)
        dataY.append(cddata[i+look_back])
    return np.array(dataX), np.array(dataY)

Model_cd = tf.keras.models.load_model('Model/train4-LOOKBACK=125(2.h5')
Model_cip = tf.keras.models.load_model('Model/train4cip-LOOKBACK=125.h5')
########################################################################

#f0 = 0.6
arr = np.arange(0, 12, 0.004, float)

def Thetabuild(x):
    data = []
    for t in range(3000):
        r1 = x[0]*sin(2*pi*x[1]*arr[t])#+x[1]*cos(2*pi*1*f0*arr[t])
        #r2 = x[2]*sin(2*pi*2*f0*arr[t])+x[3]*cos(2*pi*2*f0*arr[t])
        #r3 = x[4]*sin(2*pi*3*f0*arr[t])+x[5]*cos(2*pi*3*f0*arr[t])
        #r4 = x[6]*sin(2*pi*4*f0*arr[t])+x[7]*cos(2*pi*4*f0*arr[t])
        a =  r1 
        data.append(a)       
    return np.array(data)



def myfunc(p):
    x = p
    theta_ga = Thetabuild(x).reshape(-1,1)
    theta_ga_scaled = scaler_theta.transform(theta_ga)
    theta_ready = theta_ga_scaled[0:, :]
    theta_x, theta_y = buildDataset(theta_ready, LOOK_BACK)
    theta_x = np.reshape(theta_x, (theta_x.shape[0], 1, theta_x.shape[1]))
    ga_cipredict = Model_cip.predict(theta_x)
    ga_cipredict = scaler_cip.inverse_transform(ga_cipredict)
    ga_cdpredict = Model_cd.predict(theta_x)
    ga_cdpredict = scaler_cd.inverse_transform(ga_cdpredict) 
    cycle_step = int((4*1/(x[1]*0.004)))
    gacd = ga_cdpredict[600:600 + cycle_step]
    gacip = ga_cipredict[600:600 + cycle_step]
    scd = sum(gacd)
    scip = sum(gacip)
    return float(scd/scip)
    
ga = GA(func=myfunc, n_dim=2, max_iter=150, size_pop=50, lb=[0.17, 0.65], ub=[0.27, 0.84], precision=1e-7)
best_x, best_y = ga.run()
print('best_x:', best_x, '\n', 'best_y:', best_y)


Y_history = pd.DataFrame(ga.all_history_Y)
fig, ax = plt.subplots(2, 1)
ax[0].plot(Y_history.index, Y_history.values, '.', color='red')
Y_history.min(axis=1).cummin().plot(kind='line')
plt.show()

l = np.arange(0, 12, 0.004)
y = []
for t in l:
   #y_1 = best_x[0]*sin(2*pi*1*f0*t)+best_x[1]*cos(2*pi*1*f0*t) + best_x[2]*sin(2*pi*2*f0*t)+best_x[3]*cos(2*pi*2*f0*t)# + best_x[4]*sin(2*pi*3*f0*t)+best_x[5]*cos(2*pi*3*f0*t) + best_x[6]*sin(2*pi*4*f0*t) + best_x[7]*cos(2*pi*4*f0*t)
   y_1 = best_x[0]*sin(2*pi*1*best_x[1]*t)
   y.append(y_1)
plt.plot(l,y)