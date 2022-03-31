import tensorflow as tf
from tensorflow.keras.datasets import mnist
import numpy as np
import cvnn
from cvnn import layers
from tensorflow.keras.layers import Flatten, Conv2D, Dense, ReLU, Softmax,Activation,MaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.losses import MeanSquaredError, SparseCategoricalCrossentropy
from cvnn.losses import ComplexAverageCrossEntropy

def gpu_fix():
    '''fixes gpu issues on my pc'''
    config = tf.compat.v1.ConfigProto(gpu_options =  tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.8))
    config.gpu_options.allow_growth = True
    session = tf.compat.v1.Session(config=config)
    tf.compat.v1.keras.backend.set_session(session)

def load_mnist():
    '''load mnist  data and get it into complex shape'''
    (x_train, y_train),(x_test,y_test) = mnist.load_data()
    x_train = x_train.astype("float32")/255
    y_train = y_train.astype("float32")
    x_train = x_train[...,np.newaxis]
    
    x_test = x_test.astype("float32")/255
    x_test = x_test[...,np.newaxis]    

    # force the train and test dataset into complex values (even though it does not make sense for this dataset)
    x_train = tf.complex(x_train,x_train*3)
    y_train = tf.complex(y_train,y_train*2)
    print(x_train.d)
    return x_train,y_train,x_test,y_test

if __name__=="__main__":
    # comment this if you don't need it.
    gpu_fix() 
    # load data
    x_train,y_train,_,_ = load_mnist()    
    
    # building the model
    model = Sequential()
    # model.add(layers.ComplexConv2D(32,(3, 3), input_shape=(28, 28, 1), dtype=np.complex64))
    # model.add(Activation("cart_relu"))
    model.add(layers.ComplexFlatten(input_shape=(28, 28, 1), dtype=np.complex64))
    model.add(Activation("cart_relu"))
    model.add(layers.ComplexDense(128, dtype=np.complex64))
    model.add(Activation("cart_relu"))
    model.add(layers.ComplexDense(10, activation='cart_softmax', dtype=np.complex64))
    model.compile(loss=ComplexAverageCrossEntropy, optimizer=tf.keras.optimizers.Adam(0.0001),metrics=['accuracy'],)
    model.fit(x_train[:1000],y_train[:1000], epochs=10, shuffle=False)
    
    # predicts only the first complex value
    print("actual number:\t\t",y_train[0:1])
    result = model.predict(x_train[0:1])
    print("result is:\t\t", (result))