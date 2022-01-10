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
画图函数
""""""""""""""""""""""""""""""""""""""" 
import matplotlib.pyplot as plt
def display_digit(index):
    label = y_train[index].argmax(axis=0)
    image = X_train[index]
    plt.title('training data, index: %d, Label: %d' %(index, label))
    plt.imshow(image, cmap = 'gray_r')
    plt.show()

display_digit(0)    

"""""""""""""""""""""""""""""""""""""""
定义CNN网络结构
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
model.summary()

"""""""""""""""""""""""""""""""""""""""
训练和保存
"""""""""""""""""""""""""""""""""""""""
model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
hist = model.fit(X_train, y_train, epochs=8, verbose=2) 
#model.save_weights('Model/MNIST_12')
model.save('Model/MNIST_12')

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