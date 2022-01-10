from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.models import load_model

from tensorflow.keras import preprocessing
from tensorflow.keras import backend as K
from tensorflow.keras import models

import tensorflow as tf
import numpy as np

import matplotlib.pyplot as plt

import tensorflow as tf
import os
os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config=tf.compat.v1.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess=tf.compat.v1.Session(config=config)

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

from keras.preprocessing import image

"""""""""""""""""""""""""""""""""""""""
指定图片
""""""""""""""""""""""""""""""""""""""" 
index = 18
img = X_train[index]
x1 = image.img_to_array(img)
x2 = np.expand_dims(x1, axis=0)

"""""""""""""""""""""""""""""""""""""""
热力图
""""""""""""""""""""""""""""""""""""""" 
conv_layer = model.get_layer("conv2")
heatmap_model = models.Model([model.inputs], [conv_layer.output, model.output])

# Get gradient of the winner class w.r.t. the output of the (last) conv. layer
with tf.GradientTape() as gtape:
    conv_output, predictions = heatmap_model(x2)
    loss = predictions[:, np.argmax(predictions[0])]
    grads = gtape.gradient(loss, conv_output)
    pooled_grads = K.mean(grads, axis=(0, 1, 2))

heatmap = tf.reduce_mean(tf.multiply(pooled_grads, conv_output), axis=-1)
heatmap = np.maximum(heatmap, 0)
max_heat = np.max(heatmap)
if max_heat == 0:
    max_heat = 1e-10
heatmap /= max_heat

"""""""""""""""""""""""""""""""""""""""
热力图与原图分览
""""""""""""""""""""""""""""""""""""""" 

import cv2
original_img= img
heatmap1 = cv2.resize(heatmap[0,:,:],(28,28))

img2 = img[:,:,0]

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
ax.matshow(heatmap[0,:,:])
ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
ax.matshow(x2[0,:,:,0])
plt.show()



plt.figure(figsize=(4,4))
plt.imshow(heatmap1,cmap='CMRmap_r')
plt.axis('off')
plt.savefig('./CAM/index_h',dpi=7)

plt.imshow(img2,cmap='gray_r')
plt.axis('off')
plt.savefig('./CAM/index',dpi=7)
"""""""""""""""""""""""""""""""""""""""
热力图与原图结合
""""""""""""""""""""""""""""""""""""""" 

from PIL import Image
import matplotlib.pyplot as plt


img1 = Image.open("./CAM/index_h.png")
img1 = img1.convert('RGBA')
img2 = Image.open("./CAM/index.png")
img2 = img2.convert('RGBA')
acm = Image.blend(img1, img2, 0.8)

#plt.figure(figsize=(50, 50))
plt.imshow(acm)
plt.axis('off')
plt.show()

fig = plt.figure(figsize=(8,4))
ax = fig.add_subplot(1, 2, 1, xticks=[], yticks=[])
ax.imshow(heatmap1,cmap='CMRmap_r')
ax.axis('off')
ax = fig.add_subplot(1, 2, 2, xticks=[], yticks=[])
ax.imshow(img2,cmap='gray_r')
ax.axis('off')
plt.show()