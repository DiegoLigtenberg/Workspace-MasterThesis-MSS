from tensorflow.keras import backend as K
import numpy as np
import tensorflow as tf
import os
import time
import glob
import re

# module imports
# from mss.models.auto_encoder import AutoEncoder # for vocal sep
from mss.models.auto_encoder_other import AutoEncoder
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

def load_fsdd(spectrograms_path):
  x_train = []
  y_train = []
  sub = ["mixture","vocals","bass","drums","other","accompaniment"]

  filelist = glob.glob(os.path.join("G:/Thesis/"+spectrograms_path+"/mixture", '*'))
  filelist.sort(key=natural_keys)
  print(len(filelist))
  for i, file in enumerate(filelist):
    if i >=0 and i < 400: # remove this when full dataset
      normalized_spectrogram = np.load(file)
      x_train.append(normalized_spectrogram)
  filelist = glob.glob(os.path.join("G:/Thesis/"+spectrograms_path+"/mixture", '*'))
  filelist.sort(key=natural_keys)
  for i, file in enumerate(filelist):
    if i >=0 and i <400: # remove this when full dataset
      normalized_spectrogram = np.load(file)
      y_train.append(normalized_spectrogram)
  x_train = np.array(x_train)
  y_train = np.array(y_train)      
  return x_train,y_train


def train(learning_rate,batch_size,epochs,model_name=""):

  """ vocal model
  variatonal_auto_encoder = AutoEncoder(
      input_shape=(2048, 128, 1),
      conv_filters=(64, 128, 256, 128), # how many kernels you want per layer
      conv_kernels=(4, 4, 4, 4), # KERNEL SIZE SHOULD BE DIVISIBLE BY STRIDE! but only when upsampling! -> OTHERWISE CITY BLOCK PATTERN -> receptive field
      conv_strides=(2, 2, 2, 2), # probably also remove large stride size in beginning! UNET ENDS WITH 1x1 CONV BLOCK!
      latent_space_dim=128)
  """
  ## all models + bn model
  variatonal_auto_encoder = AutoEncoder(
      input_shape=(2048, 128, 1),
      conv_filters=(64, 64, 128, 128, 256, 512), # how many kernels you want per layer
      conv_kernels=(5,   5,   3,   3,   3,   3), # KERNEL SIZE SHOULD BE DIVISIBLE BY STRIDE! but only when upsampling! -> OTHERWISE CITY BLOCK PATTERN -> receptive field (at how big grid it looks)
      conv_strides=(2,   2,   2,   2,   2,   2), # probably also remove large stride size in beginning! UNET ENDS WITH 1x1 CONV BLOCK!
      latent_space_dim=128)
    
    # variatonal_auto_encoder = AutoEncoder(
  #     input_shape=(2048, 128, 1),
  #     conv_filters=(64, 64, 128, 128, 256, 512), # how many kernels you want per layer
  #     conv_kernels=(3,   3,   3,   3,   3,   3), # KERNEL SIZE SHOULD BE DIVISIBLE BY STRIDE! but only when upsampling! -> OTHERWISE CITY BLOCK PATTERN -> receptive field
  #     conv_strides=(2,   2,   2,   2,   2,   2), # probably also remove large stride size in beginning! UNET ENDS WITH 1x1 CONV BLOCK!
  #     latent_space_dim=128)
  
  # variatonal_auto_encoder = AutoEncoder(
  #     input_shape=(2048, 128, 1),
  #     conv_filters=(64, 64,   128,  128,   256,  256, 256, 512), # how many kernels you want per layer
  #     conv_kernels=(7,   3,    7,    3,     3,    3,   3,   3), # KERNEL SIZE SHOULD BE DIVISIBLE BY STRIDE! but only when upsampling! -> OTHERWISE CITY BLOCK PATTERN -> receptive field
  #     conv_strides=(2,   1,    2,    1,     2,    2,   2,   2), # probably also remove large stride size in beginning! UNET ENDS WITH 1x1 CONV BLOCK!
  #     latent_space_dim=128)

  variatonal_auto_encoder.name = model_name
  variatonal_auto_encoder.summary()
  variatonal_auto_encoder.compile(learning_rate)
  variatonal_auto_encoder.train_on_batch(batch_size,epochs) #y_train
  return variatonal_auto_encoder


def main():
    
    # the more complex the model -> the lower the lr should be
    BATCH_SIZE = 8
    LEARNING_RATE = 1e-4
    EPOCHS = 201
    MODEL_NAME = "model_a_low_val" #"model_instruments_other-10-0.00635"

    ''' first training'''
    # for i in range(0,1):
      # variational_auto_encoder = AutoEncoder.load("Final_Model_Other-10-0.01871-0.03438 VALID") 
      # variational_auto_encoder = train(learning_rate=LEARNING_RATE,batch_size=BATCH_SIZE,epochs=EPOCHS,model_name=MODEL_NAME)   #0.003  
      # variational_auto_encoder.save(MODEL_NAME)
    # 0.001 is already decent-ish !!!!! 
    # print(5/0)
     

    '''repeated training'''  
    # print("new learnn rate:",LEARNING_RATE)
    # for i in range(1):
    #   variational_auto_encoder = AutoEncoder.load("model_otherVOCAL_BN_mae-27-0.01822-0.03593") 
    #   variational_auto_encoder.name="model_otherVOCAL_BN_mae"
    #   variational_auto_encoder.compile(learning_rate=LEARNING_RATE)       
    #   variational_auto_encoder.train_on_batch(BATCH_SIZE,EPOCHS)    
    #   variational_auto_encoder.save("model_otherVOCAL_BN_mae")

    '''repeated training'''  
    # print("new learnn rate:",LEARNING_RATE)
    # for i in range(1):
      # variational_auto_encoder = AutoEncoder.load("model_a_low_val-5-0.00272-0.00468") 
      # variational_auto_encoder.name="model_a_low_val"
      # variational_auto_encoder.compile(learning_rate=LEARNING_RATE)       
      # variational_auto_encoder.train_on_batch(BATCH_SIZE,EPOCHS)    
      # variational_auto_encoder.save("model_otherVOCAL_BN_mae")

      

    '''repeated training'''  
    print("new learnn rate:",LEARNING_RATE)
    for i in range(1):
      variational_auto_encoder = AutoEncoder.load("Final_Model_Other_extra_songs-120-0.0148-0.03536") 
      variational_auto_encoder.name="Final_Model_Other_extra_songs"
      variational_auto_encoder.compile(learning_rate=LEARNING_RATE)       
      variational_auto_encoder.train_on_batch(BATCH_SIZE,EPOCHS)    
      variational_auto_encoder.save("Final_Model_Other_extra_songs")


      

if __name__=="__main__":
    main()