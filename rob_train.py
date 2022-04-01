from enum import auto
# from aa import VariationalAutoEncoder
from auto_ancoder_lolstaatdaarmssnietvoorxD import VariationalAutoEncoder
from tensorflow.keras.datasets import mnist
import numpy as np
import tensorflow as tf
import os
import time
import glob
import re

LEARNING_RATE = 0.00005  #mnist 0.0001 
BATCH_SIZE = 8
EPOCHS = 30
LOAD_SPECTROGRAMS_PATH = "train"

#https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

def load_mnist():
  # 60.000 training  -  10.000 testing
  (x_train, y_train),(x_test,y_test) = mnist.load_data()
  x_train = x_train.astype("float32")/255
  x_train = x_train[...,np.newaxis]
  
  x_test = x_test.astype("float32")/255
  x_test = x_test[...,np.newaxis]

  return x_train,y_train,x_test,y_test

def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]


def load_fsdd(spectrograms_path):
  x_train = []
  y_train = []

  filelist = glob.glob(os.path.join("train/mixture", '*'))
  filelist.sort(key=natural_keys)
  for i, file in enumerate(filelist):
    if i <100:
      # print(file) 
      normalized_spectrogram = np.load(file)
      x_train.append(normalized_spectrogram)
    else:
      break
  filelist = glob.glob(os.path.join("train/mixture", '*'))
  filelist.sort(key=natural_keys)
  for i, file in enumerate(filelist):
    if i <100:
      # print(file) 
      normalized_spectrogram = np.load(file)
      y_train.append(normalized_spectrogram)
    else:
      break
    # print(file)
  x_train = np.array(x_train)
  y_train = np.array(y_train)      
  print(x_train.shape)
  print(y_train.shape)
  return x_train,y_train

def train(x_train,y_train,learning_rate,batch_size,epochs):
  # MNIST
  # variatonal_auto_encoder = VariationalAutoEncoder(
  #   input_shape=(28, 28, 1),
  #   conv_filters=(32, 64, 64, 64),
  #   conv_kernels=(3, 3, 3, 3),
  #   conv_strides=(1, 2, 2, 1),
  #   latent_space_dim=2)

  variatonal_auto_encoder = VariationalAutoEncoder(
    input_shape=(2048, 128, 1),
    conv_filters=(32, 64, 64, 64),
    conv_kernels=(3, 3, 3, 3), # KERNEL SIZE SHOULD BE DIVISIBLE BY STRIDE! but only when upsampling! -> OTHERWISE CITY BLOCK PATTERN
    conv_strides=(4, 2, 2, 2), # probably also remove large stride size in beginning! UNET ENDS WITH 1x1 CONV BLOCK!
    latent_space_dim=128)

  variatonal_auto_encoder.summary()
  variatonal_auto_encoder.compile(learning_rate)
  variatonal_auto_encoder.train(x_train,x_train,batch_size,epochs) #y_train
  return variatonal_auto_encoder


from tensorflow.keras import backend as K
if __name__=="__main__":
    x_train,y_train = load_fsdd(LOAD_SPECTROGRAMS_PATH) 
    variational_auto_encoder = train(x_train[:30],x_train[:30],0.001,BATCH_SIZE,EPOCHS)
    LEARNING_RATE = 0.001 
    variational_auto_encoder.compile(learning_rate=LEARNING_RATE)
    variational_auto_encoder.train(x_train[:30],x_train[:30],BATCH_SIZE,EPOCHS*3)
    variational_auto_encoder.save("model_skipcon")