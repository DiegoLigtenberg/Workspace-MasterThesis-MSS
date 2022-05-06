from audioop import minmax
from math import prod
from mss.preprocessing.preprocesssing import MinMaxNormalizer

import numpy as np
import matplotlib.pyplot as plt
# from auto_encoder_vanilla import VariationalAutoEncoder
# from mss.models.auto_encoder import AutoEncoder
from mss.models.auto_encoder_other import AutoEncoder
from mss.models.atrain import load_fsdd
import librosa
import librosa.display
from scipy.io import wavfile
from scipy.signal import wiener
import tensorflow as tf
from tensorflow.keras import backend as K
from mss.postprocessing.generator_c import *

from mss.utils.dataloader import natural_keys, atof
import glob
import os
import pickle
'''
For each track in track_input

1) Load track
  - load IRMAS DATABASE
  - load MUSDB DATABASE
2) preprocess the track
  - preprocess IRMAS DATABASE
  - (MUSDB already preprocessed)
3) Model the track
  - convert preprocessed data into waveforms (track wise)
  - name the generated tracks according to original track_names
4) Evaluate tracks
'''

class EncodeTestData():
    def __init__(self, save=False):
        self.save = save
        self.filelist_X, self.filelist_Y = self.load_musdb()

    def load_musdb(self):
        def save_pickle():
            filelist_X = glob.glob(os.path.join("G:/Thesis/test/mixture", '*'))
            filelist_X.sort(key=natural_keys)
            filelist_X = filelist_X[0::]
            filelist_Y = glob.glob(os.path.join("G:/Thesis/test/other", '*'))
            filelist_Y.sort(key=natural_keys)
            filelist_Y = filelist_Y[0::]

            filelist_X_new = []
            file_list_Y_new = []

            current_song = -1
            for i, song in enumerate(filelist_X):
                chunks_X = []
                chunks_Y = []
                if current_song != int(filelist_X[i].split("\\")[1].split("-")[0]):
                    current_song = int(
                        filelist_X[i].split("\\")[1].split("-")[0])
                    for chunk in filelist_X:
                        if int(chunk.split("\\")[1].split("-")[0]) == current_song:
                            chunks_X.append(chunk)
                    for chunk in filelist_Y:
                        if int(chunk.split("\\")[1].split("-")[0]) == current_song:
                            chunks_Y.append(chunk)
                    filelist_X_new.append(chunks_X)
                    file_list_Y_new.append(chunks_Y)
            with open('mss_evaluate_data/test_X.pkl', 'wb') as f:
                print("files saved to:\t", "mss_evaluate_data")
                pickle.dump(filelist_X_new, f)
            with open('mss_evaluate_data/test_Y.pkl', 'wb') as f:
                pickle.dump(file_list_Y_new, f)
        if self.save:
            save_pickle()
        filelist_X = pickle.load(open("mss_evaluate_data/test_X.pkl", "rb"))
        filelist_Y = pickle.load(open("mss_evaluate_data/test_Y.pkl", "rb"))
        return filelist_X, filelist_Y

    def evaluate(self):
        '''evaluates each song in musdb test database'''
        # Split the songs in test dataset in songs -> chunks
        for song_x, song_y in zip(self.filelist_X, self.filelist_Y):
            print(len(song_x))
       

            x_train = []
            y_train = []

            for i, file in enumerate(song_x):
                normalized_spectrogram = np.load(file)
                x_train.append(normalized_spectrogram)

            for i, file in enumerate(song_y):
                normalized_spectrogram = np.load(file)
                y_train.append(normalized_spectrogram)

            x_train = np.array(x_train)
            y_train = np.array(y_train)
            break
        return x_train, y_train

class Generate():
    '''takes as input x_mixture , y_other'''
    
    def __init__(self) -> None:
        self.auto_encoder = AutoEncoder.load("Final_Model_Other-10-0.01871-0.03438 VALID") 
        self.filelist_X,self.filelist_Y = EncodeTestData(save=False).load_musdb()

        self.song_iterator = -1
        # song is 1 lower than when you start songs counting from 1 !
        self.file_length = len(self.filelist_X)
        self.gen()
        print("done")

    def gen_eval(self):
        '''generate and evaluate based on sdr'''
        # for song_x, song_y in zip(self.filelist_X, self.filelist_Y):
        while self.song_iterator < self.file_length:
            self.song_iterator+=1
            # yield all the chunks (mixture,target) of a test song
            yield self.filelist_X[self.song_iterator], self.filelist_Y[self.song_iterator] 

    def gen(self):
        gen = Generator()
        for i in range(self.file_length):    
            x_mixture_file_chunks,y_target_file_chunks = next(self.gen_eval())                
            gen.generate_waveform(x_mixture_file_chunks,y_target_file_chunks,self.song_iterator,inference=False,save_mixture=True)


gener = Generate()
