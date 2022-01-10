import  pandas as pd
import  numpy as np
import tensorflow as tf
import keras
import matplotlib.pyplot as plt


from keras.layers.recurrent import LSTM
from sklearn.preprocessing import MinMaxScaler

from keras.models import Sequential

from tensorflow.keras.layers import Dense, Lambda, Dot, Activation, Concatenate
from tensorflow.keras.layers import Layer
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config=tf.compat.v1.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess=tf.compat.v1.Session(config=config)

"""""""""""""""""""""""""""""""""""""""
定义Attention层
"""""""""""""""""""""""""""""""""""""""
class Attention(Layer):

    def __init__(self, units=128, **kwargs):
        self.units = units
        super().__init__(**kwargs)

    def __call__(self, inputs):
        hidden_states = inputs
        hidden_size = int(hidden_states.shape[2])
        score_first_part = Dense(hidden_size, use_bias=False, name='attention_score_vec')(hidden_states)
        h_t = Lambda(lambda x: x[:, -1, :], output_shape=(hidden_size,), name='last_hidden_state')(hidden_states)
        score = Dot(axes=[1, 2], name='attention_score')([h_t, score_first_part])
        attention_weights = Activation('softmax', name='attention_weight')(score)
        context_vector = Dot(axes=[1, 1], name='context_vector')([hidden_states, attention_weights])
        pre_activation = Concatenate(name='attention_output')([context_vector, h_t])
        attention_vector = Dense(self.units, use_bias=False, activation='tanh', name='attention_vector')(pre_activation)
        return attention_vector

    def get_config(self):
        return {'units': self.units}

    @classmethod
    def from_config(cls, config):
        return cls(**config)



"""""""""""""""""""""""""""""""""""""""
定义生成LSTM数据集格式函数
"""""""""""""""""""""""""""""""""""""""
def create_dataset(dataset, look_back):
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i+look_back),:]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    Train_X = np.array(dataX)
    Train_Y = np.array(dataY)
    return Train_X, Train_Y



"""""""""""""""""""""""""""""""""""""""
读取数据
"""""""""""""""""""""""""""""""""""""""

train_rowdata = pd.read_csv('./dataset/train/train4pure.csv')
train_cipdata = (train_rowdata['Cip'].values).reshape(len(train_rowdata),1)
train_cddata = (train_rowdata['Cd'].values).reshape(len(train_rowdata),1)
train_thetadata = (train_rowdata['theta'].values).reshape(len(train_rowdata),1)
train_attdata = np.array(train_rowdata.drop(['time','real_moment','omega','Cip'], axis = 1))
test_rowdata = pd.read_csv('./dataset/test/15_0.8Hz.csv')
test_cipdata = (test_rowdata['Cip'].values).reshape(len(test_rowdata),1)
test_cddata = (test_rowdata['Cd'].values).reshape(len(test_rowdata),1)
test_thetadata = (test_rowdata['theta'].values).reshape(len(test_rowdata),1)
test_attdata = np.array(test_rowdata.drop(['time','real_moment','omega','Cip','forcex','moment'], axis = 1))
test_attdata[:,[0,1]] = test_attdata[:,[1,0]] #交换顺序使theta在前，Cd在后


"""""""""""""""""""""""""""""""""""""""
以训练集为基准定义归一化
"""""""""""""""""""""""""""""""""""""""


scaler_theta = MinMaxScaler(feature_range=(-1,1))
scaler_theta.fit(train_thetadata[:,0].reshape(-1,1))
train_theta_scaled = scaler_theta.transform(train_thetadata[:,0].reshape(-1,1))
test_theta_scaled = scaler_theta.transform(test_thetadata[:,0].reshape(-1,1))

scaler_cd = MinMaxScaler(feature_range=(-1,1))
scaler_cd.fit(train_cddata[:,0].reshape(-1,1))
train_cd_scaled = scaler_cd.transform(train_cddata[:,0].reshape(-1,1))
test_cd_scaled = scaler_cd.transform(test_cddata[:,0].reshape(-1,1))

scaler_cip = MinMaxScaler(feature_range=(-1,1))
scaler_cip.fit(train_cipdata)
train_cip_scaled = scaler_cip.transform(train_cipdata)
test_cip_scaled = scaler_cip.transform(test_cipdata)


"""""""""""""""""""""""""""""""""""""""
生成LSTM的数据集
"""""""""""""""""""""""""""""""""""""""

Motion_history = 125 #运动的历史步长，水流流过翼型特征长度的时间/CFD计算时间步长
train_THETA, _ = create_dataset(train_theta_scaled,Motion_history)
_ , train_CD = create_dataset(train_cd_scaled,Motion_history)
_ , train_CIP = create_dataset(train_cip_scaled,Motion_history)

train_THETA = np.reshape(train_THETA, (train_THETA.shape[0], train_THETA.shape[1], 1))


"""""""""""""""""""""""""""""""""""""""
定义神经网络模型结构
"""""""""""""""""""""""""""""""""""""""

Model = Sequential([
      LSTM(4, input_shape=(Motion_history, 1), activation='tanh',return_sequences=True),
      LSTM(2, input_shape=(Motion_history, 1), activation='tanh',return_sequences=True),
      Attention(8,name='attention_weight'),
      Dense(4,activation='relu'),
      Dense(1)
])
my_op = keras.optimizers.RMSprop(learning_rate=0.005)

Model.compile(loss='mse', optimizer=my_op)
history = Model.fit(train_THETA, train_CIP, epochs=100, verbose=2)
# print(Model.summary())
Model.save_weights('Model/duelingQ_CIP_100_9_14.h5')


plt.plot(history.history['loss'])
plt.xlim(200,500)
plt.ylim(0,0.009)
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")
plt.show()

plt.plot(history.history['loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")
plt.show()

# """""""""""""""""""""""""""""""""""""""
# 加载神经网络模型
# """""""""""""""""""""""""""""""""""""""



# # Model = Model.load('Model/Cd_CNN_time_att_5_1000.h5')

# """""""""""""""""""""""""""""""""""""""
# 训练集查看
# """""""""""""""""""""""""""""""""""""""

# train_predict_cip = Model.predict(train_THETA)
# train_CIP = np.reshape(train_CIP, (train_CIP.shape[0]))
# train_predict_cip = scaler_cip.inverse_transform(train_predict_cip)
# train_CIP = scaler_cip.inverse_transform([train_CIP])

# train_predict_plot = np.empty_like(train_cipdata)
# train_predict_plot[:,:] = np.nan
# train_predict_plot[Motion_history:len(train_predict_cip) + Motion_history, :] = train_predict_cip
# plt.figure(figsize=(20, 7))
# plt.plot(train_cipdata,label='original')
# plt.xlabel('time-step', fontsize=20)
# plt.ylabel('CIP', fontsize=20)
# plt.title('Training-set',fontsize=30)
# plt.plot(train_predict_plot,label='predict')
# plt.legend(loc=0, numpoints=1)
# leg = plt.gca().get_legend()
# ltext = leg.get_texts()
# plt.setp(ltext, fontsize=20)
# plt.show()
# train_CIP_pure = train_CIP[0]
# train_score = math.sqrt(mean_squared_error(train_CIP[0,100:], train_predict_cip[100:,0])) #为了消除启动的干扰,往后顺延一百步
# print(f'训练误差分数（RMSE）：{np.round(train_score, 3)}')

# """""""""""""""""""""""""""""""""""""""
# 测试集查看
# """""""""""""""""""""""""""""""""""""""

# test_THETA, _ = create_dataset(test_theta_scaled,Motion_history)
# _ , test_CD = create_dataset(test_cd_scaled,Motion_history)
# _ , test_CIP = create_dataset(test_cip_scaled,Motion_history)
# test_THETA = np.reshape(test_THETA, (test_THETA.shape[0], test_THETA.shape[1], 1))

# #test_prex = np.concatenate((test_theta_scaled,test_cip_scaled),axis = 1)
# #test_ATTX, _ = create_dataset(test_prex,Motion_history)
# #test_ATTX = np.reshape(test_ATTX, (test_ATTX.shape[0], test_ATTX.shape[1], Dim))

# test_predict = Model.predict(test_THETA)
# test_predict = scaler_cip.inverse_transform(test_predict)

# test_predict_plot = np.empty_like(test_cipdata)
# test_predict_plot[:,:] = np.nan
# test_predict_plot[Motion_history:len(test_predict) + Motion_history, :] = test_predict


# plt.figure(figsize=(20, 7))
# plt.plot(test_cipdata,label='original')

# plt.plot(test_predict_plot)
# plt.xlabel('timestep', fontsize=20)
# plt.ylabel('CIP', fontsize=20)
# plt.title('Test-data',fontsize=30)
# plt.plot(test_predict_plot,label='predict')
# plt.legend(loc=0, numpoints=1)
# leg = plt.gca().get_legend()
# ltext = leg.get_texts()
# plt.setp(ltext, fontsize=20)
# plt.show()
# test_score = math.sqrt(mean_squared_error(test_cipdata[226:],test_predict[100:]))#为了消除启动的干扰,往后顺延一百步
# print(f'测试误差分数（RMSE）：{np.round(test_score, 3)}')


# """""""""""""""""""""""""""""""""""""""
# 计算周期平均推力系数
# """""""""""""""""""""""""""""""""""""""

# T = 1.25 #运动周期
# t = 0.004 #计算时间步
# Cippredict_T = test_predict[300: 300 + int(T/t)]
# Cip_T = test_cipdata[300: 300 + int(T/t)]
# f_1 = sum(Cippredict_T)*t/T
# print(f'预测Cd值：{np.round(f_1, 3)}')
# f_2 = sum(Cip_T)*t/T
# print(f'真实Cd值:{np.round(f_2, 3)}')
