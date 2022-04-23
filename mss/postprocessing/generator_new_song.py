from audioop import minmax
from math import prod
from mss.preprocessing.preprocesssing import MinMaxNormalizer

import numpy as np
import matplotlib.pyplot as plt
# from auto_encoder_vanilla import VariationalAutoEncoder
# from mss.models.auto_encoder import AutoEncoder
from mss.models.auto_encoder_other import AutoEncoder
from mss.models.atrain import load_fsdd
import librosa, librosa.display
from scipy.io import wavfile
from scipy.signal import wiener
import tensorflow as tf
from tensorflow.keras import backend as K

from mss.utils.dataloader import natural_keys, atof
import glob
import os

'''
For each track in track_input

1) Load track
2) preprocess the track 
3) Model the track

'''



def load_input_track(spectrograms_path):
#   sub = ["mixture","vocals","bass","drums","other","accompaniment"]

  filelist = glob.glob(os.path.join("track_input", '*'))
  filelist.sort(key=natural_keys)
  print(len(filelist))

  # 
  for i, file in enumerate(filelist):
    if i >=0 and i < 400: # remove this when full dataset
      normalized_spectrogram = np.load(file)
      x_train.append(normalized_spectrogram)
  filelist = glob.glob(os.path.join("F:/Thesis/"+spectrograms_path+"/other", '*'))
  filelist.sort(key=natural_keys)
  for i, file in enumerate(filelist):
    if i >=0 and i <400: # remove this when full dataset
      normalized_spectrogram = np.load(file)
      y_train.append(normalized_spectrogram)
  x_train = np.array(x_train)
  y_train = np.array(y_train)      
  return x_train,y_train