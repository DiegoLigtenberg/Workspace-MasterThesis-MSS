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

           
            # self.song_iterator+=1

            # for song_x, song_y in zip(self.filelist_X, self.filelist_Y):

        #     x_train,y_train = next("test") # note the amnt of datapoints load_fssd loads -> check the function
        # pass

    def gen(self):
        gen = Generator()
        for i in range(self.file_length):    
            x_mixture_file_chunks,y_target_file_chunks = next(self.gen_eval())                
            gen.generate_waveform(x_mixture_file_chunks,y_target_file_chunks,self.song_iterator)
        # x_mixture,y_target = next(self.gen(self.filelist_X[self.song_iterator],self.filelist_Y[self.song_iterator]))
        
        
        # yield x_mixture_file_chunks, y_target_file_chunks

        # while self.song_iterator < 
        '''generate only'''

        pass

    
    
    

    total_track = []
    reall = False
    estimate = None
    reference = None

    '''
    for song in self.file_list_X:
      for chunk in song:
        if save:
          #saving waveforms from generator
          chunk_wav_estimate = generator.get_waveform()
          chunk_wav_target = generator.get_waveform()
        else:
          #load
          # chunk_wav = open("waveform_domain")
        
        sdr = chunk_wav
        mean_sdr = np.mean(chunk_sdr)
    '''
        # generate song
        # print(data)
        # pass


gener = Generate()
# print(app.filelist_X[0])

# def load_input_track(spectrograms_path):
# #   sub = ["mixture","vocals","bass","drums","other","accompaniment"]

#   filelist = glob.glob(os.path.join("track_input", '*'))
#   filelist.sort(key=natural_keys)
#   print(len(filelist))

#   #
#   for i, file in enumerate(filelist):
#     if i >=0 and i < 400: # remove this when full dataset
#       normalized_spectrogram = np.load(file)
#       x_train.append(normalized_spectrogram)
#   filelist = glob.glob(os.path.join("F:/Thesis/"+spectrograms_path+"/other", '*'))
#   filelist.sort(key=natural_keys)
#   for i, file in enumerate(filelist):
#     if i >=0 and i <400: # remove this when full dataset
#       normalized_spectrogram = np.load(file)
#       y_train.append(normalized_spectrogram)
#   x_train = np.array(x_train)
#   y_train = np.array(y_train)
#   return x_train,y_train

'''

auto_encoder = AutoEncoder.load("Final_Model_Other-10-0.01871-0.03438 VALID")  #model_spectr for first_source_sep
auto_encoder.summary()
b_train,y_train = load_fsdd("test") # note the amnt of datapoints load_fssd loads -> check the function
(np.min(b_train),np.max(b_train))


total_track = []
reall = False
for r in range (3):
    # r=2
    total_track = []
    for i in range(108,109,1): # test 140-160 should be very good! [8, 56, 112, 216, 312, 560]
        sound = i #132 test

        # weights = np.full_like(b_train[:1],1/prod(b_train[:1].shape))
        # test[0][:512]*=3
        # test[0][1536:]/=3
        # print(test[0][:512])
        # # print(test)
        # print(5/0)
        if r  <=1:
            x_train=np.array(y_train[sound:sound+1]) # y_train when vocal source sep
        else:
            x_train=np.array(b_train[sound:sound+1]) 

        # x_train += (np.random.rand(b_train.shape[0],b_train.shape[1],b_train.shape[2],b_train.shape[3])-0.5) * 0.3
        print(i,x_train.shape)
        if r == 0:
            x_train = auto_encoder.model.predict(b_train[sound:sound+1])

        x_val, y_val = b_train[sound:sound+1],y_train[sound:sound+1]
        y_pred = x_train
        y_pred = tf.convert_to_tensor(y_pred,dtype=tf.float32)
        y_val = tf.cast(y_val, y_pred.dtype)
        val_loss = K.mean(tf.math.squared_difference(y_pred, y_val), axis=-1)
        val_loss = np.mean(val_loss.numpy())
        
        print("error\t\t",val_loss)
        print("error\t\t",np.mean(np.abs((x_train[:1]-y_train[sound:sound+1])**2)))
        print("min and max val:",np.min(x_train),np.max(x_train))
        print("mean:\t\t",np.mean(x_train))

        mute_sound = False
        if  -0.15 < np.min(x_train) < 0.15 and -0.15 < np.max(x_train) < 0.15 and -0.15 < np.mean(x_train) < 0.15:
            print("mute sound")
            mute_sound = True

        error = (x_train-y_train[sound:sound+1]) *5# *5 to exagerate
        # x_train +=error
        
        # plt.imshow(error[0],cmap="gray",vmin=-1,vmax=1)
        # plt.show()

        print(x_train.shape)
        # print(min(x_train))
        # print(max(x_train))
        min_max_normalizer = MinMaxNormalizer(0,1)   
        
        x_train = min_max_normalizer.denormalize(x_train)
        x_train = x_train [:,:,:,0]
        # print(x_train[0] == x_train[1])
        x_train = x_train[0]
        x_train = x_train[:,:127]
        x_train = x_train[:-1]   
        # x_train[500:] =0 
        x_train = librosa.db_to_amplitude(x_train) 
        
        amp_log_spectrogram = librosa.amplitude_to_db(x_train,ref=np.max)
        fig, ax = plt.subplots()      
        img = librosa.display.specshow(amp_log_spectrogram, y_axis='linear', sr=44100, hop_length=1050,  x_axis='time', ax=ax)
        ax.set(title='Log-amplitude spectrogram')
        ax.label_outer()
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        plt.show()

        # x_train = librosa.db_to_amplitude(x_train) 
        # x_source = wiener(x_train, (5, 5))
        # print(x_source.shape)
        # scale = lambda x: x*1.5 
        # scale(x_train)

        # original phase ( gets lot of noise dont do it)
        # signal,sr = librosa.load("original.wav",sr=44100)
        # stft = librosa.stft(signal,n_fft=4096,hop_length=1024)[:-2]     
        # mag,phase = librosa.magphase(stft)
        # phase = phase[:,:127]   
        # print(phase.shape)
        # print(x_train.shape)
        # new_stft = x_train +1j*phase
        # print(new_stft.shape)
    
        x_source = librosa.griffinlim(x_train,hop_length=1050)
        if mute_sound:
            x_source = np.zeros_like(x_source)+0.001
        # x_source*=1.5
        print((x_source.shape))
        total_track.append(x_source)
        # print(x_source)
        print("\n\n\n")
        # print(x_train.shape)
        # print(x_source.shape)
    total_track = np.array(total_track)
    total_track = total_track.flatten()
    
    print((total_track.shape))
    if r == 0:
        total_track = wiener(total_track,mysize=3)
        wavfile.write("track_output/other_predict.wav",44100,total_track) 
    elif r == 1:            
        wavfile.write("track_output/other_target.wav",44100,total_track) 
    else:
        wavfile.write("track_output/other_mixture.wav",44100,total_track) 

'''
