from enum import auto

from tensorflow.keras.datasets import mnist
from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
import os
import time
import glob
import re

# module imports
from mss.models.auto_encoder import AutoEncoder
from mss.utils.dataloader import natural_keys, atof
# from auto_encoder_vanilla import VariationalAutoEncoder

LEARNING_RATE = 0.00005  #mnist 0.0001 
BATCH_SIZE = 8
EPOCHS = 2
LOAD_SPECTROGRAMS_PATH = "G:/Thesis"
# tf.keras.initalizer

#https://stackoverflow.com/questions/43147983/could-not-create-cudnn-handle-cudnn-status-internal-error
physical_devices = tf.config.experimental.list_physical_devices('GPU')
assert len(physical_devices) > 0, "Not enough GPU hardware devices available"
config = tf.config.experimental.set_memory_growth(physical_devices[0], True)

initializer = tf.keras.initializers.TruncatedNormal(mean=0., stddev=1.)
layer = tf.keras.layers.Dense(3, kernel_initializer=initializer)

def load_mnist():
  # 60.000 training  -  10.000 testing
  (x_train, y_train),(x_test,y_test) = mnist.load_data()
  x_train = x_train.astype("float32")/255
  x_train = x_train[...,np.newaxis]
  
  x_test = x_test.astype("float32")/255
  x_test = x_test[...,np.newaxis]

  return x_train,y_train,x_test,y_test

def load_fsdd(spectrograms_path):
  x_train = []
  y_train = []
  sub = ["mixture","vocals","bass","drums","other","accompaniment"]
  c= 0
  '''
  # THIS LOOP IS NOT WOrKING!!! MULTIPLE TIMES LOADING SAME FILE
  '''

  filelist = glob.glob(os.path.join("G:/Thesis/"+spectrograms_path+"/mixture", '*'))
  filelist.sort(key=natural_keys)
  print(len(filelist))
  for i, file in enumerate(filelist):
    if i >=0 and i < 200: # remove this when full dataset
      # print(file) 
      normalized_spectrogram = np.load(file)
      x_train.append(normalized_spectrogram)
    # else:
    #   break
  filelist = glob.glob(os.path.join("G:/Thesis/"+spectrograms_path+"/vocals", '*'))
  filelist.sort(key=natural_keys)
  for i, file in enumerate(filelist):
    if i >=0 and i <200: # remove this when full dataset
      # print(file) 
      normalized_spectrogram = np.load(file)
      y_train.append(normalized_spectrogram)
    # else:
    #   break
    # print(file)
  x_train = np.array(x_train)
  y_train = np.array(y_train)      
  # print(x_train.shape)
  # print(y_train.shape)
  return x_train,y_train

  ''' 
  for root,_,file_names in os.walk(spectrograms_path):
    for file_name in file_names:
      file_path = os.path.join(root,file_name)
      #mixture
      if sub[0] in file_path:
        print(file_name)
        normalized_spectrogram = np.load(file_path)
        # normalized_spectrogram = normalized_spectrogram[:-1]
        # normalized_spectrogram = np.hstack((normalized_spectrogram,np.zeros_like(normalized_spectrogram)[::]))
        # print(normalized_spectrogram.shape)
        # for i in range(126):
        #   normalized_spectrogram = np.delete(normalized_spectrogram,1,axis=1)

        # normalized_spectrogram = np.insert(normalized_spectrogram,0,np.zeros_like(normalized_spectrogram[:,0]))
        # print(normalized_spectrogram.shape)
        # print(5/0)
        x_train.append(normalized_spectrogram)
        c+=1
        if c>500:
          break
      #vocals
      # if sub[1] in file_path:
      #   normalized_spectrogram = np.load(file_path)
      #   y_train.append(normalized_spectrogram)
  c= 0
  for root,_,file_names in os.walk(spectrograms_path):
    for file_name in file_names:
      file_path = os.path.join(root,file_name)
      #vocals
      if sub[1] in file_path:
        normalized_spectrogram = np.load(file_path)
        y_train.append(normalized_spectrogram)
        c+=1
        if c>500:
          break
  x_train = np.array(x_train)
  y_train = np.array(y_train)      
  print(x_train.shape)
  print(y_train.shape)
  return x_train,y_train
'''

def train(x_train,y_train,learning_rate,batch_size,epochs):
  # MNIST
  # variatonal_auto_encoder = VariationalAutoEncoder(
  #   input_shape=(28, 28, 1),
  #   conv_filters=(32, 64, 64, 64),
  #   conv_kernels=(3, 3, 3, 3),
  #   conv_strides=(1, 2, 2, 1),
  #   latent_space_dim=2)

  # working settings
    # variatonal_auto_encoder = AutoEncoder(
    # input_shape=(2048, 128, 1),
    # conv_filters=(32, 64, 64, 64),
    # conv_kernels=(3, 3, 3, 3), # KERNEL SIZE SHOULD BE DIVISIBLE BY STRIDE! but only when upsampling! -> OTHERWISE CITY BLOCK PATTERN
    # conv_strides=(2, 2, 2, 2), # probably also remove large stride size in beginning! UNET ENDS WITH 1x1 CONV BLOCK!
    # latent_space_dim=128)

  # very good one
  # variatonal_auto_encoder = AutoEncoder(
  #   input_shape=(2048, 128, 1),
  #   conv_filters=(128, 128, 256, 128), # how many kernels you want per layer
  #   conv_kernels=(4, 4, 4, 8), # KERNEL SIZE SHOULD BE DIVISIBLE BY STRIDE! but only when upsampling! -> OTHERWISE CITY BLOCK PATTERN -> receptive field
  #   conv_strides=(2, 2, 2, 2), # probably also remove large stride size in beginning! UNET ENDS WITH 1x1 CONV BLOCK!
  #   latent_space_dim=128)

  # model_train_on_batch_vocals batch size = 8 lr = 5e-7
  # variatonal_auto_encoder = AutoEncoder(
  #     input_shape=(2048, 128, 1),
  #     conv_filters=(128, 128, 256, 128), # how many kernels you want per layer
  #     conv_kernels=(4, 4, 4, 4), # KERNEL SIZE SHOULD BE DIVISIBLE BY STRIDE! but only when upsampling! -> OTHERWISE CITY BLOCK PATTERN -> receptive field
  #     conv_strides=(2, 2, 2, 2), # probably also remove large stride size in beginning! UNET ENDS WITH 1x1 CONV BLOCK!
  #     latent_space_dim=128)

  variatonal_auto_encoder = AutoEncoder(
      input_shape=(2048, 128, 1),
      conv_filters=(64, 128, 256, 128), # how many kernels you want per layer
      conv_kernels=(4, 4, 4, 4), # KERNEL SIZE SHOULD BE DIVISIBLE BY STRIDE! but only when upsampling! -> OTHERWISE CITY BLOCK PATTERN -> receptive field
      conv_strides=(2, 2, 2, 2), # probably also remove large stride size in beginning! UNET ENDS WITH 1x1 CONV BLOCK!
      latent_space_dim=128)

    # the more complex the model -> the lower the lr should be

  variatonal_auto_encoder.summary()
  variatonal_auto_encoder.compile(learning_rate)
  variatonal_auto_encoder.train(x_train,y_train,batch_size,epochs) #y_train
  return variatonal_auto_encoder


def main():
      # x_train,_,_,_ = load_mnist()    
    # variational_auto_encoder = train(x_train[:5000],LEARNING_RATE,BATCH_SIZE,EPOCHS)
    # variational_auto_encoder.save("model_gentest")

    # load data    
    x_train,y_train = load_fsdd(LOAD_SPECTROGRAMS_PATH) 
    BATCH_SIZE = 8
    LEARNING_RATE = 3e-4
    EPOCHS = 70


    # first training
    # for i in range(0,1):
    #   variational_auto_encoder = train(x_train[:1],y_train[:1],LEARNING_RATE,batch_size=BATCH_SIZE,epochs=EPOCHS)   #0.003 
    #   variational_auto_encoder.save("model_train_on_batch_vocals3")
    # # # 0.02 is already decent-ish !!!!! 
    # # print(5/0)
     
    LEARNING_RATE = 6e-5
      
    # print("new learnn rate:",LEARNING_RATE)
    BATCH_SIZE = 8
    print(LEARNING_RATE)
    for i in range(1):
      variational_auto_encoder = AutoEncoder.load("model_train_on_batch_vocals3-19-9995.0")   
      variational_auto_encoder.compile(learning_rate=LEARNING_RATE)       
      variational_auto_encoder.train_on_batch(BATCH_SIZE,EPOCHS)    
      variational_auto_encoder.save("model_train_on_batch_vocals3-final")
      # LEARNING_RATE/=2
      # BATCH_SIZE=1
      # LEARNING_RATE = 3e-4

    # LEARNING_RATE = 5e-3 #5e-7 
    # repeated training
    # for i in range (3):
    #   variational_auto_encoder = AutoEncoder.load("model_train_on_batch_vocals2")    
      
      
    #   print("new learnn rate:",LEARNING_RATE)
    #   BATCH_SIZE = 8
    #   variational_auto_encoder.compile(learning_rate=LEARNING_RATE)   
    
    #   variational_auto_encoder.train(x_train[:1],y_train[:1],BATCH_SIZE,EPOCHS//2)    
    #   variational_auto_encoder.save("model_train_on_batch_vocals2")
    #   LEARNING_RATE/=2
    # 4 epoch


    # variational_auto_encoder.save("model_skipcon")
    # variational_auto_encoder.save("model_skipconrob20")
    # variational_auto_encoder = train(x_train[:1],x_train[:1],0.001,1,EPOCHS*40)
    # variational_auto_encoder = VariationalAutoEncoder.load("model_skipconrob20")
    # LEARNING_RATE = 0.0001 
    # variational_auto_encoder.save("model_skipconrob20")
    # asd
    
    # print(np.min(x_train),np.max(x_train[:1]))
    # variational_auto_encoder.train(x_train[:1],x_train[:1],1,EPOCHS*1)
    # variational_auto_encoder.save("model_vocal_sep")
    # variational_auto_encoder = VariationalAutoEncoder.load("model_spectr")
    # 0.0635 loss
    # variational_auto_encoder = VariationalAutoEncoder.load("model_spectr")
    # LEARNING_RATE = 0.00005 
    # variational_auto_encoder = train(x_train,x_train,LEARNING_RATE,BATCH_SIZE,EPOCHS)
    # variational_auto_encoder.save("model_spectr")

    # 0.0635 loss


if __name__=="__main__":
    main()