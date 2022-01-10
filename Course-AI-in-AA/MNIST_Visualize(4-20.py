"""""""""""""""""""""""""""""""""""""""
引入MNIST数据集
"""""""""""""""""""""""""""""""""""""""
import tensorflow as tf
import numpy as np
from keras.datasets import mnist

(X_train, y_train), (X_test, y_test) = mnist.load_data() 
X_train = np.reshape(X_train, [-1,28,28,1])
X_test = np.reshape(X_test, [-1,28,28,1])

from keras.utils.np_utils import to_categorical
y_train = to_categorical(y_train, 10)
y_test = to_categorical(y_test, 10)
 
"""""""""""""""""""""""""""""""""""""""
加载训练好的模型
""""""""""""""""""""""""""""""""""""""" 
  
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, Activation

model = Sequential([
    Conv2D(filters=16, kernel_size=2, padding='same', activation='relu', input_shape=(28,28,1),name='conv1'),
    MaxPooling2D(pool_size=2,name='maxpool1'),
    Conv2D(filters=32, kernel_size=2, padding='same', activation='relu',name='conv2'),
    MaxPooling2D(pool_size=2,name='maxpool2'),
    Dropout(0.2),
    Flatten(),
    Dense(10),
    Activation('softmax')
    ])
model.load_weights('Model/MNIST_10')
#model.summary()

"""""""""""""""""""""""""""""""""""""""
可视化中间层
""""""""""""""""""""""""""""""""""""""" 
from keras.preprocessing import image

index = 1
img = X_train[index]
x1 = image.img_to_array(img)
x2 = np.expand_dims(x1, axis=0)
x3 = np.expand_dims(img, axis=0)

#img = np.reshape(img,[-1,28,28,1])
#plt.imshow(img)
import matplotlib.pyplot as plt
from tensorflow.keras import models


visualize_model = models.Model(inputs=model.input, outputs=model.get_layer('conv1').output)#输出为conv1层的输出 
conv1_outputs = visualize_model.predict(img)

fig = plt.figure(figsize=(8,8))
for i in range(16):
    ax = fig.add_subplot(4, 4, i+1, xticks=[], yticks=[])
    ax.matshow(conv1_outputs[:, :, :, i], cmap='gray_r')
    #label = y_train[index].argmax(axis=0)
    #plt.title('Label: %d' %(label))





