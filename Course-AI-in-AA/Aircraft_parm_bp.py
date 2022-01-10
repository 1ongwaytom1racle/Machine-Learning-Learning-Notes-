import  pandas as pd
import  numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from tensorflow.keras.layers import Dense
"""""""""""""""""""""""""""""""""""""""
读取数据
"""""""""""""""""""""""""""""""""""""""

train_rowdata = np.array(pd.read_csv('./dataset/Aircraft_Parm/traindata.csv'))
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
#预测后四列的值
train_x = trainset_scaled[:,0:5]
train_y = trainset_scaled[:,6:10]
test_x = testset_scaled[:,0:5]
test_y = testset_scaled[:,6:10]

"""""""""""""""""""""""""""""""""""""""
定义神经网络模型结构
"""""""""""""""""""""""""""""""""""""""
"""
Model = Sequential([
      Dense(12,activation = 'tanh'),
      Dense(12,activation = 'tanh'),
      Dense(4, activation = 'sigmoid')
])

Model.compile(loss='mse', optimizer='adam')
history = Model.fit(train_x, train_y, epochs=2000, verbose=0)
print(Model.summary())
Model.save('Model/Aircraft_parm_bp(3.h5')
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

Model = tf.keras.models.load_model('Model/Aircraft_parm_bp(3.h5')


"""""""""""""""""""""""""""""""""""""""
测试集查看
"""""""""""""""""""""""""""""""""""""""

#测试集反归一化
def inver_scale(test_predict):
    data = []
    for i in range(4):
        scaler = scaler_maker(i+6)[2]
        a = scaler.inverse_transform(np.reshape(test_predict[:,i],(1,-1)))
        data.append(a)
    data = np.reshape(np.array(data),(4,-1)).T 
    return data

test_predict = Model.predict(test_x) 
test_predict = inver_scale(test_predict)    

"""""""""""""""""""""""""""""""""""""""
画图
"""""""""""""""""""""""""""""""""""""""

name_list = ['DC9','A320','A310','B747']
fig = plt.figure(figsize=(10, 8))
plt.subplot(4,1,1)  
plt.scatter(name_list, test_predict[:,0], marker='+', s=150)
plt.scatter(name_list, test_rowdata[:,6], s=150)
plt.xticks(())
plt.ylabel('Gco2')
plt.subplot(4,1,2)  
plt.scatter(name_list, test_predict[:,1], marker='+', s=150)
plt.scatter(name_list, test_rowdata[:,7], s=150)
plt.xticks(())
plt.ylabel('Gch4')
plt.subplot(4,1,3)  
plt.scatter(name_list, test_predict[:,2], marker='+', s=150)
plt.scatter(name_list, test_rowdata[:,8], s=150)
plt.ylabel('Gacid')
plt.xticks(())
plt.subplot(4,1,4)  
plt.scatter(name_list, test_predict[:,3], marker='+', s=150)
plt.scatter(name_list, test_rowdata[:,9], s=150)
plt.ylabel('Gtoxic')
plt.show()













   