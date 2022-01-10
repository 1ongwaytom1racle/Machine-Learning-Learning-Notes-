from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from tensorflow.keras.layers import Lambda, Input, Dense
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Conv1D, Reshape, Conv1DTranspose
from tensorflow.keras.losses import mse
from tensorflow.keras import backend as K
import keras
import numpy as np
import matplotlib.pyplot as plt
import os
import tensorflow as tf

os.environ["CUDA_VISIBLE_DEVICES"] = '0' #use GPU with ID=0
config=tf.compat.v1.ConfigProto() 
config.gpu_options.per_process_gpu_memory_fraction = 0.5 # maximun alloc gpu50% of MEM
config.gpu_options.allow_growth = True #allocate dynamically
sess=tf.compat.v1.Session(config=config)

# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(args):

    z_mean, z_log_var = args
    # K is the keras backend
    batch = K.shape(z_mean)[0]
    dim = K.int_shape(z_mean)[1]
    # by default, random_normal has mean=0 and std=1.0
    epsilon = K.random_normal(shape=(batch, dim))
    return z_mean + K.exp(0.5 * z_log_var) * epsilon


tri_data = np.load('tri_data_1000.npy')
rect_data = np.load('rect_data_1000.npy')
sin_data = np.load('sin_data_1000.npy')
tri_data = np.vstack([tri_data,rect_data,sin_data])
tri_data[:,:,0] /= 200  #稍微归一化
tri_data[:,:,1] /= 200  

# # network parameters
input_shape = (tri_data.shape[1], tri_data.shape[2])
intermediate_dim = 256
batch_size = 64
latent_dim = 2
epochs = 10
kernel_size = 2
filters = 16

inputs = Input(shape=input_shape, name='encoder_input')
x = inputs
for i in range(2):
    filters *= 2
    x = Conv1D(filters=filters,
               kernel_size=kernel_size,
               activation='relu',
               strides=2,
               padding='same')(x)
shape = K.int_shape(x)
x1 = Flatten()(x)
x2 = Dense(intermediate_dim, activation='relu')(x1)
x3 = Dense(2*intermediate_dim, activation='relu')(x2)
z_mean = Dense(latent_dim, name='z_mean')(x3)
z_log_var = Dense(latent_dim, name='z_log_var')(x3)

z = Lambda(sampling,
            output_shape=(latent_dim,), 
            name='z')([z_mean, z_log_var])

# # instantiate encoder model
encoder = Model(inputs, [z_mean, z_log_var, z], name='encoder')
encoder.summary()

shuru = tri_data[0:1,:,:]
aaa = encoder(shuru)

# # build decoder model
latent_inputs = Input(shape=(latent_dim,), name='z_sampling')
x = Dense(2*intermediate_dim,activation='relu')(x)
x = Dense(intermediate_dim,activation='relu')(x)
x = Dense(shape[1] * shape[2],
          activation='relu')(latent_inputs)
x = Reshape((shape[1], shape[2]))(x)


filters = filters/2
x = Conv1DTranspose(filters=filters,
                    kernel_size=kernel_size,
                    activation='relu',
                    strides=2,
                    padding='same')(x)

outputs = Conv1DTranspose(filters=2,
                          kernel_size=kernel_size,
                          # activation='sigmoid',
                          padding='same',
                          strides=2,
                          name='decoder_output')(x)
decoder = Model(latent_inputs, outputs, name='decoder')
decoder.summary()

# instantiate VAE model
outputs = decoder(encoder(inputs)[2])
vae = Model(inputs, outputs, name='vae_mlp')
aaaa = vae(shuru)
aaaa = np.array(aaaa)

models = (encoder, decoder)

# VAE loss = mse_loss or xent_loss + kl_loss
reconstruction_loss = mse(K.flatten(inputs),
                                          K.flatten(outputs))
reconstruction_loss *= tri_data.shape[1]* tri_data.shape[2]
#     reconstruction_loss *= original_dim
kl_loss = 1 + z_log_var - K.square(z_mean) - K.exp(z_log_var)
kl_loss = K.sum(kl_loss, axis=-1)
kl_loss *= -0.5
vae_loss = K.mean(reconstruction_loss + kl_loss)
vae.add_loss(vae_loss)
my_op = keras.optimizers.Adam(learning_rate=0.00005)
vae.compile(optimizer = my_op)
vae.fit(tri_data,
        epochs=epochs,
        batch_size=batch_size)
vae.save_weights('12_13_deep_vae')

xmin = ymin = -5
xmax = ymax = +5


# display a 2D plot of the digit classes in the latent space
z, _, _ = encoder.predict(tri_data,
                          batch_size=32) #此处z是z_mean
plt.figure(figsize=(12, 10))

# # axes x and y ranges
axes = plt.gca()
axes.set_xlim([xmin,xmax])
axes.set_ylim([ymin,ymax])

# subsample to reduce density of points on the plot
z = z[0::10]
y_test = tri_data[0::10]
plt.xlabel("z[0]")
plt.ylabel("z[1]")
# plt.show()


n = 30
digit_size = 28
figure = np.zeros((digit_size * n, digit_size * n))
# linearly spaced coordinates corresponding to the 2D plot
# of digit classes in the latent space
grid_x = np.linspace(-4, 4, n)
grid_y = np.linspace(-4, 4, n)[::-1]
plot_ratio = 0.5#缩放比例系数
for i in range(16):
    for j in range(16):
        z_sample = np.array([[0.5*(i - 8),0.5*(j - 8)]])
        x_decoded = decoder.predict(z_sample)
        plt.plot(plot_ratio*(x_decoded[0,:,0]) + 0.5*(i - 8),
                 plot_ratio*(x_decoded[0,:,1]) + 0.5*(j - 8))

