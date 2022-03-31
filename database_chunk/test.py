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

config = tf.compat.v1.ConfigProto(gpu_options =  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
config.gpu_options.allow_growth = True
session = tf.compat.v1.Session(config=config)
tf.compat.v1.keras.backend.set_session(session)

def load_mnist():
    # 60.000 training  -  10.000 testing
    (x_train, y_train),(x_test,y_test) = mnist.load_data()
    x_train = x_train.astype("float32")/255
    y_train = y_train.astype("float32")
    x_train = x_train[...,np.newaxis]
    
    x_test = x_test.astype("float32")/255
    x_test = x_test[...,np.newaxis]    
    x_train = tf.concat(x_train,x_train*3)
    y_train = tf.concat(y_train,y_train*2)
    return x_train,y_train,x_test,y_test

x_train,y_train,_,_ = load_mnist()

model = Sequential()
model.add(layers.ComplexConv2D(32,(3, 3), input_shape=(28, 28, 1), dtype=np.float32))
model.add(Activation("relu"))
model.add(layers.ComplexFlatten(input_shape=(28, 28, 1), dtype=np.float32))
model.add(Activation("relu"))
model.add(layers.ComplexDense(128, dtype=np.float32))
model.add(Activation("relu"))
model.add(layers.ComplexDense(10, activation='softmax', dtype=np.float32))

model.compile(loss='sparse_categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(0.0001),metrics=['accuracy'],)


model.fit(x_train[:2000],y_train[:2000], epochs=10, shuffle=False)
result = (model.predict(x_train[0:1]))
print("actual number:\t\t",y_train[0:1])
print(np.argmax(result,axis=1))
# print("predicted number:\t",np.argmax(result,axis=1))

# print(result)
# print(y_train[0])