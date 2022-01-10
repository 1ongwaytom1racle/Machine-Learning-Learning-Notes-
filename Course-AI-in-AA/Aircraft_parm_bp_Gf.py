import  pandas as pd
import  numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from tensorflow.keras.layers import Dense
from neupy import algorithms
"""""""""""""""""""""""""""""""""""""""
读取数据
"""""""""""""""""""""""""""""""""""""""

train_rowdata = np.array(pd.read_csv('./dataset/Aircraft_Parm/traindata.csv'))#此处为数据的路径
test_rowdata = np.array(pd.read_csv('./dataset/Aircraft_Parm/testdata.csv'))

"""""""""""""""""""""""""""""""""""""""
以训练集为基准归一化
"""""""""""""""""""""""""""""""""""""""

#输入列数，返回三维元组（归一化后的训练列，测试列，归一化器）
def scaler_maker (i):
    scaler = MinMaxScaler()
    scaler.fit(train_rowdata[:,i].reshape(-1,1))
    train_scaled = scaler.transform(train_rowdata[:,i].reshape(-1,1))
    test_scaled = scaler.transform(test_rowdata[:,i].reshape(-1,1))
    return(train_scaled, test_scaled, scaler)
       
#输入0返回归一化训练集，输入1返回归一化测试集 
def scale_data(t):
    data = []
    for i in range(10):
        a = scaler_maker(i)[t]
        data.append(a)
    data = np.reshape(np.array(data),(10,-1)).T 
    return data

trainset_scaled = scale_data(0)
testset_scaled = scale_data(1)
#预测第六列的值
train_x = trainset_scaled[:,0:5]
train_y = trainset_scaled[:,5]
test_x = testset_scaled[:,0:5]
test_y = testset_scaled[:,5]
"""""""""""""""""""""""""""""""""""""""
定义神经网络模型结构
"""""""""""""""""""""""""""""""""""""""
"""
Model = Sequential([
      Dense(10,activation = 'tanh'),
      Dense(10,activation = 'tanh'),
      Dense(1,activation = 'sigmoid')
])

Model.compile(loss='mse', optimizer='adam')
history = Model.fit(train_x, train_y, epochs=2000, verbose=0)
print(Model.summary())
Model.save('Model/Aircraft_parm_bp_Gf(3.h5')

plt.plot(history.history['loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")
plt.show()
"""
"""""""""""""""""""""""""""""""""""""""
加载神经网络模型
"""""""""""""""""""""""""""""""""""""""

Model = tf.keras.models.load_model('Model/Aircraft_parm_bp_Gf(3.h5')


"""""""""""""""""""""""""""""""""""""""
测试机查看
"""""""""""""""""""""""""""""""""""""""

#测试集反归一化
def inver_scale(test_predict):
    data = []
    i = 0
    scaler = scaler_maker(i+5)[2]
    a = scaler.inverse_transform(np.reshape(test_predict[:,i],(1,-1)))
    data.append(a)
    data = np.reshape(np.array(data),(4,-1)).T 
    return data

test_predict = Model.predict(test_x) 
test_predict = inver_scale(test_predict).T    

"""""""""""""""""""""""""""""""""""""""
画图
"""""""""""""""""""""""""""""""""""""""

name_list = ['DC9','A320','A310','B747']
fig = plt.figure(figsize=(10, 8))  
plt.scatter(name_list, test_predict[:,0], marker='+', s=150)
plt.scatter(name_list, test_rowdata[:,5], s=150)
plt.ylabel('Gf')
plt.show()





