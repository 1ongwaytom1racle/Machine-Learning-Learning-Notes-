"""""""""""""""""""""""""""""""""""""""
引入MNIST数据集
"""""""""""""""""""""""""""""""""""""""
import tensorflow as tf
import numpy as np
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data() 
X_train = np.reshape(X_train, [-1,28,28,1])
X_test = np.reshape(X_test, [-1,28,28,1])

def circle_class(x):#按有无环形结构分组
    for i in range(len(x)):
         if  x[i] == 0 or x[i] == 4 or x[i] == 6 or x[i] == 8 or x[i] == 9:
              x[i] = 0              
         else:
             x[i] = 1
              
circle_class(y_train)
circle_class(y_test)

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, 2)
y_test = to_categorical(y_test, 2)

"""""""""""""""""""""""""""""""""""""""
加载训练好的模型
""""""""""""""""""""""""""""""""""""""" 
  
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

model_M = Sequential([
    Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1),name='conv1'),
    MaxPooling2D(pool_size=2,name='maxpool1'),
    Conv2D(filters=32, kernel_size=2, padding='same', activation='relu',name='conv2'),
    MaxPooling2D(pool_size=2,name='maxpool2'),
    Dropout(0.2),
    Flatten(),
    Dense(10),
    Activation('softmax')
    ])
model_M.load_weights('Model/MNIST_10')
#model.summary()

"""""""""""""""""""""""""""""""""""""""
迁移学习保留卷积层
""""""""""""""""""""""""""""""""""""""" 

import matplotlib.pyplot as plt
from tensorflow.keras import models


tl_model = models.Model(inputs=model_M.input, outputs=model_M.get_layer('maxpool2').output)


top_model = Sequential()
top_model.add(Flatten(input_shape=tl_model.output_shape[1:]))
top_model.add(Dense(2, activation='softmax'))

model = Sequential()
model.add(tl_model)
model.add(top_model)
model.summary()


for layer in model.layers[:1]:
    layer.trainable = False

model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
hist = model.fit(X_train, y_train, epochs=5, verbose=2) 

plt.plot(hist.history['loss'])
plt.title("model loss")
plt.ylabel("loss")
plt.xlabel("epoch")
plt.legend(["train","test"],loc="upper left")
plt.show()

"""""""""""""""""""""""""""""""""""""""
测试集
""""""""""""""""""""""""""""""""""""""" 
score = model.evaluate(X_test, y_test, batch_size=32,
                       verbose=1,sample_weight=None)
print('Test score:', score[0])
print('Test accuracy:', score[1])




