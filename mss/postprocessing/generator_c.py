from audioop import minmax
from math import prod

from torch import chunk
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
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanAbsoluteError
from mss.settings.settings import *
import os
from matplotlib.pyplot import close


DATAFOLDER = "mss_evaluate_data/database_test/"
VISUALISATION_FOLDER = "mss_evaluate_data/visualisation/"

SAVE_WAVEFORM = True           # if true: saves waveform of chosen model
SAVE_LOSS = False                # if true: saves loss and visualizes it (msa loss and sdr loss)
VISUALIZE_SPECTROGRAM = False   # if true: saves spectrogram (predict/target/mixture) for each track
VERBOSE = False                 # if true: only print sdr and msa metrics when uploading
model = "_2" # empty strsing if no adjust



class Generator():
    def __init__(self):
        self.auto_encoder = AutoEncoder.load("Final_Model_Other_extra_songs-15-0.01687-0.03398 VALID")
        self.database_msa = []
        self.database_sdr = []
        # self.generate_waveform()
        # self.song_msa = np.mean(self.song_msa)
        # self.song_sdr = np.mean(self.song_sdr)

    def generate_waveform(self, x_mixture_file_chunks, y_target_file_chunks, song_number, inference=False,save_mixture=False):
        '''
        input:          a batch of numpy spectrograms
        output:         a full waveform of the batched spectrograms

        inference:      if not in inference mode mode; only generate waveform, else  generate waveform of predict and target + compute sdr and msa loss
        save_mix:       if in save_mixture mode; generate waveform of original mixture
        '''
        self.x_mixture, self.y_target = self._load_spectrograms(x_mixture_file_chunks, y_target_file_chunks)
        self.magnitude_song_chunks = len(x_mixture_file_chunks)      

        total_track = []
        self.song_msa = []
        self.song_sdr = []
        self.estimate_list = []

        # initialize setting logic
        instances = ["predict", "target", "mixture"]
        if inference and save_mixture: instances = ["predict", "mixture"]
        if inference and not save_mixture: instances = ["predict"]
        if not inference and save_mixture: instances = ["predict", "target", "mixture"]
        if not inference and not save_mixture: instances = ["predict", "target"]
        
        # iterate over each song
        for instance in instances:
            self.song_name = f"{song_number}-{instance}"
            total_track = []            
            # iterate over each 3 second exerpt in a song
            for chunk in range(self.magnitude_song_chunks):
                if instance == "predict":
                    # predict
                    x_train = self.auto_encoder.model.predict(self.x_mixture[chunk:chunk+1])   

                    # making sound fundamental and removing drums!    
                    # '''         
                    x_train[x_train<=-0.0] = -.33                   # hardforce, only use this if first fails
                    # x_train[(x_train<0.05) & (x_train >= 0.0)]  -=.025  # even more strict 
                    x_train[(x_train<0.05) & (x_train >= 0.0)]  *=.2  # the lower the multiplication number ( closer to 0) -> the more sounds are removed
                    x_train[(x_train<0.0) & (x_train > -0.2)]  /=.2   # the lower the division number (closer to 0) -> the more sound (drums) are removed, but also other sound
                    x_train[(x_train<=-0.2)] = -.33
                    # '''
                    
                    # calculate msa
                    if inference == False:
                        self.song_msa.append(self.calculate_msa(x_train,chunk))                    
                if instance == "target":
                    # target
                    x_train = np.array(self.y_target[chunk:chunk+1])
                    name = f"{chunk}\t{self.song_name}-{chunk}"
                    if VERBOSE: print(name,end="\r")
                if instance == "mixture":
                    # mixture
                    x_train = np.array(self.x_mixture[chunk:chunk+1])
                    name = f"{chunk}\t{self.song_name}-{chunk}"
                    if VERBOSE: print(name,end="\r")

                # mute sound if there is full silence (spectrogram util)
                mute_sound = False
                if -0.15 < np.min(x_train) < 0.15 and -0.15 < np.max(x_train) < 0.15 and -0.15 < np.mean(x_train) < 0.15:
                    if VERBOSE: print("mute chunk")
                    mute_sound = True

                # 1) min max normalizing 
                min_max_normalizer = MinMaxNormalizer(0, 1)
                x_train = min_max_normalizer.denormalize(x_train)

                # 2) fixing dimensions with input
                x_train = x_train[:, :, :, 0]
                x_train = x_train[0]
                # remove last timestep that was added with 0's
                x_train = x_train[:, :127]
                # add 1 more frequency bin that is same as last frequency
                x_train = np.vstack((x_train, x_train[-1]))
                x_train = librosa.db_to_amplitude(x_train)
        
                # visualize spectrogram if needed
                if VISUALIZE_SPECTROGRAM:
                    self.visualize_spectrogram(x_train,instance,song_number,chunk)

                # 3) calculate estimate and reference spectrogram
                if instance == "predict":
                    self.estimate_list.append(x_train)
                if instance == "target":
                    reference = x_train
                    self.song_sdr.append(self.calculate_sdr(reference,chunk))

                # 4) generate waveform
                x_source = librosa.griffinlim(x_train,hop_length=HOP_LENGTH)

                # 5) fix waveform when silence
                if mute_sound:
                    x_source = np.zeros_like(x_source)+0.001 # +0.001 s.t no noise is heard                    
               
                # 6) add waveform chunk to total track to total_track
                assert x_source.shape[0] == SAMPLE_RATE * CHUNK_DURATION, f"chunk is too short {x_source.shape}"
                total_track.append(x_source)

            # 7) combine all waveform chunks into final track
            total_track = np.array(total_track)
            total_track = total_track.flatten()

            # 8) save wavefile
            if instance == "predict":
                self.save_wave_form(instance,total_track)                
            elif instance == "target":
                # calculate song_sdr and song_msa with reference and estimate 
                self.song_sdr = np.mean(self.song_sdr)                
                self.song_msa = np.mean(self.song_msa) 
                
                # calculate loss
                self.calculate_loss(self.song_msa,self.song_sdr)
                if SAVE_LOSS:                    
                    self.visualize_loss() # only update visualize when saving
                self.save_wave_form(instance,total_track)
                
            else:
                if instance == "mixture":
                    self.save_wave_form(instance,total_track)

    def save_wave_form(self,instance,total_track):
        if SAVE_WAVEFORM:
            folder = f"{DATAFOLDER}{instance}"
            if not os.path.exists(folder): os.makedirs(folder)
            wavfile.write(f"{folder}/{self.song_name}.wav", SAMPLE_RATE, total_track)

    def visualize_spectrogram(self,x_train,instance,song_number,chunk):
        amp_log_spectrogram = librosa.amplitude_to_db(x_train, ref=np.max)
        fig, ax = plt.subplots()
        img = librosa.display.specshow(amp_log_spectrogram, y_axis='linear', sr=SAMPLE_RATE, hop_length=HOP_LENGTH,  x_axis='time', ax=ax)
        ax.set(title='Log-amplitude spectrogram')
        ax.label_outer()
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        folder = f"{VISUALISATION_FOLDER}spectrograms/{instance}/{song_number}/"
        if not os.path.exists(folder): os.makedirs(folder)
        fig.savefig(f"{folder}{chunk}")
        plt.close()

    def calculate_loss(self,song_msa,song_sdr):
            print(f"song {self.song_name.split('-')[0]}\t{round(song_msa,5)}")
            print(f"song {self.song_name.split('-')[0]}\t{round(song_sdr,5)}")
            self.database_msa.append(song_msa)
            self.database_sdr.append(song_sdr)
            folder = f"{DATAFOLDER}"
            if SAVE_LOSS:
                np.save(f"{folder}database_msa{model}",np.array(self.database_msa))
                np.save(f"{folder}database_sdr{model}",np.array(self.database_sdr))
            self.song_msa = []
            self.song_sdr = []
            self.estimate_list = []

    def visualize_loss(self):
        x_msa = np.load(f"{DATAFOLDER}database_msa{model}.npy")
        x_sdr = np.load(f"{DATAFOLDER}database_sdr{model}.npy")
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(x_msa, label='msa loss')
        ax.plot(np.ones_like(x_msa) * np.mean(x_msa), label = 'avg msa')
        ax.set(title='MSA loss per song in MUSDB test')
        plt.xlabel("test song")
        plt.ylabel("MSA")
        ax.legend()
        folder = f"{VISUALISATION_FOLDER}loss_metrics/"
        if not os.path.exists(folder): os.makedirs(folder)
        fig.savefig(f"{folder}msa_loss{model}")
        close(fig)
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(x_sdr, label='sdr loss')
        ax.plot(np.ones_like(x_sdr)*np.mean(x_sdr), label = 'avg sdr')
        ax.set(title='SDR loss per song in MUSDB test')
        plt.xlabel("test song")
        plt.ylabel("SDR")
        ax.legend()
        folder = f"{VISUALISATION_FOLDER}loss_metrics/"
        if not os.path.exists(folder): os.makedirs(folder)
        fig.savefig(f"{folder}sdr_loss{model}")
        close(fig)
    
    def calculate_msa(self,x_train,chunk):        
        y_targ = self.y_target[chunk:chunk+1]
        msa = MeanAbsoluteError()
        msa = msa(x_train, y_targ).numpy()        
        name = f"{chunk}\t{self.song_name}-{chunk}\terror\t"
        if VERBOSE: print(name,"\t",msa,end="\r")
        return msa

    def calculate_sdr(self,reference,chunk):
        delta = 1e-7  # avoid numerical errors
        # 0 is column, 1 is row
        num = np.sum(np.square(reference), axis=(0, 1))
        den = np.sum(np.square(reference - self.estimate_list[chunk]), axis=(0, 1))
        num += delta
        den += delta
        sdr = 10 * np.log10(num / den)
        name = f"{chunk}\t{self.song_name}-{chunk}\sdr\t"
        if VERBOSE: print(name,"\t",sdr,end="\r")
        return sdr   

   

    def _load_spectrograms(self, x_mixture_file_chunks, y_target_file_chunks):
        x_mixture = []
        y_target = []
        assert len(x_mixture) < 400, f"program does not support individual input tracks longer than 20 minutes"
        for i, file in enumerate(x_mixture_file_chunks):
            normalized_spectrogram = np.load(file)
            x_mixture.append(normalized_spectrogram)
        for i, file in enumerate(y_target_file_chunks):
            normalized_spectrogram = np.load(file)
            y_target.append(normalized_spectrogram)
        x_mixture = np.array(x_mixture)
        y_target = np.array(y_target)
        return x_mixture, y_target