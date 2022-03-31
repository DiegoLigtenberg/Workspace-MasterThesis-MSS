import musdb
import os
import librosa
import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import cvnn
from cvnn import layers
from tensorflow.keras.layers import Flatten, Conv2D, Dense, ReLU, Softmax,Activation,MaxPooling2D
from tensorflow.keras.models import Sequential
# tensorflow.test.is_gpu_available() 
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from cvnn.losses import ComplexAverageCrossEntropy


mus_train = musdb.DB("database_wav",subsets="train", split='train',download=False,is_wav=True)
def iterate_tracks(tracks): # this functions stops when it yields the values 
    for i,track in enumerate(tracks):
        if i < 1:
            track.chunk_duration = 10.0
            max_chunks = int(track.duration/track.chunk_duration)
            for j in range (0,max_chunks):
                track.chunk_start = j * track.chunk_duration 
                x = (track.audio) # don't transpose it
                y = (track.targets["vocals"].audio)
                x = x[:,0]
                y = y[:,0]
                x = librosa.core.stft(x,hop_length=1024,n_fft=4096,center=True)
                y = librosa.core.stft(y,hop_length=1024,n_fft=4096,center=True)
                x = x[...,np.newaxis]    
                y = y[...,np.newaxis]
                print(x.shape)
                # print(i)
            
    return x,y

x_train,y_train = iterate_tracks(mus_train)

model = Sequential()
model.add(layers.ComplexConv2D(32,(3, 3), input_shape=(2049, 431, 1), dtype=np.float32))
model.add(Activation("relu"))
# model.add(layers.ComplexFlatten(input_shape=(2049, 431, 1), dtype=np.float32))
# model.add(Activation("relu"))
# model.add(layers.ComplexDense(128, dtype=np.float32))
# model.add(Activation("relu"))
# model.add(layers.ComplexDense(10, activation='softmax', dtype=np.float32))

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0001),metrics=['accuracy'],)


model.fit(x_train[:2000],y_train[:2000], epochs=10, shuffle=False)
result = (model.predict(x_train[0:1]))
print("actual number:\t\t",y_train[0:1])
print(np.argmax(result,axis=1))
iterate_tracks(mus_train)
