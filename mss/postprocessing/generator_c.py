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
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError
from mss.settings.settings import *
import os

VISUALIZE = False
DATAFOLDER = "mss_evaluate_data/database_test/"


class Generator():
    def __init__(self, calculate_loss=False):
        self.auto_encoder = AutoEncoder.load("Final_Model_Other-10-0.01871-0.03438 VALID")
        self.database_mse = []
        self.database_sdr = []
        # self.generate_waveform()
        # self.song_mse = np.mean(self.song_mse)
        # self.song_sdr = np.mean(self.song_sdr)

    def generate_waveform(self, x_mixture_file_chunks, y_target_file_chunks, song_number, eval=False):
        '''
        input:  a batch of numpy spectrograms
        output: a full waveform of the batched spectrograms

        eval:   if in eval mode; generate waveform of predict and target, compute sdr and mse loss 
        '''
        self.x_mixture, self.y_target = self._load_spectrograms(x_mixture_file_chunks, y_target_file_chunks)
        self.magnitude_song_chunks = len(x_mixture_file_chunks)      

        total_track = []
        self.song_mse = []
        self.song_sdr = []
        self.estimate_list = []
        instances = ["predict", "target", "mixture"]
        # only predict
        if eval:
            instances = [instances[0]]

        # iterate over each song
        for instance in instances:
            self.song_name = f"{song_number}-{instance}"
            total_track = []
            
            for chunk in range(self.magnitude_song_chunks):
                if instance == instances[0]:
                    # predict
                    x_train = self.auto_encoder.model.predict(self.x_mixture[chunk:chunk+1])
                    y_targ = self.y_target[chunk:chunk+1]
                    mse = MeanSquaredError()
                    mse = mse(x_train, y_targ).numpy()
                    self.song_mse.append(mse)
                    name = f"{chunk}\t{self.song_name}-{chunk}\terror\t"
                    print(name,"\t",mse,end="\r")
                if instance == instances[1]:
                    # target
                    # y_target when vocal source sep
                    x_train = np.array(self.y_target[chunk:chunk+1])
                    name = f"{chunk}\t{self.song_name}-{chunk}"
                    print(name,end="\r")
                if instance == instances[2]:
                    # mixture
                    x_train = np.array(self.x_mixture[chunk:chunk+1])
                    name = f"{chunk}\t{self.song_name}-{chunk}"
                    print(name,end="\r")

                mute_sound = False
                if -0.15 < np.min(x_train) < 0.15 and -0.15 < np.max(x_train) < 0.15 and -0.15 < np.mean(x_train) < 0.15:
                    print("mute chunk")
                    mute_sound = True

                # min max normalizing
                min_max_normalizer = MinMaxNormalizer(0, 1)
                x_train = min_max_normalizer.denormalize(x_train)

                # fixing dimensions with input
                x_train = x_train[:, :, :, 0]
                x_train = x_train[0]
                # remove last timestep that was added with 0's
                x_train = x_train[:, :127]
                # add 1 more frequency bin that is same as last frequency
                x_train = np.vstack((x_train, x_train[-1]))
                x_train = librosa.db_to_amplitude(x_train)
        
                if VISUALIZE:
                    amp_log_spectrogram = librosa.amplitude_to_db(
                        x_train, ref=np.max)
                    fig, ax = plt.subplots()
                    img = librosa.display.specshow(
                        amp_log_spectrogram, y_axis='linear', sr=44100, hop_length=1050,  x_axis='time', ax=ax)
                    ax.set(title='Log-amplitude spectrogram')
                    ax.label_outer()
                    fig.colorbar(img, ax=ax, format="%+2.f dB")
                    plt.show()

                # calculate estimate and reference spectrogram
                if instance == instances[0]:
                    self.estimate_list.append(x_train)
                    # estimate = x_train
                if instance == instances[1]:
                    reference = x_train

                    # defossez calculation mean std (new sdr) -> causes division by zero at padded signals -> bad cant use
                    # reference2 = (reference - np.mean(reference)) / np.std(reference)
                    # self.estimate_list[chunk] = (self.estimate_list[chunk] - np.mean(reference)) / np.std(reference)
                    # reference = reference2

                    delta = 1e-7  # avoid numerical errors
                    # 0 is column, 1 is row
                    num = np.sum(np.square(reference), axis=(0, 1))
                    den = np.sum(np.square(reference - self.estimate_list[chunk]), axis=(0, 1))
                    num += delta
                    den += delta
                    sdr = 10 * np.log10(num / den)
                    name = f"{chunk}\t{self.song_name}-{chunk}\sdr\t"
                    print(name,"\t",sdr,end="\r")
                    self.song_sdr.append(sdr)
               

                # generate waveform
                x_source = librosa.griffinlim(x_train,hop_length=HOP_LENGTH)

                # fix waveform when silence
                if mute_sound:
                    # +0.001 s.t no noise is heard
                    x_source = np.zeros_like(x_source)+0.001

                # 132300 = 3 second excerpt
                assert x_source.shape[0] == SAMPLE_RATE * \
                    CHUNK_DURATION, f"chunk is too short {x_source.shape}"
                total_track.append(x_source)

            total_track = np.array(total_track)
            total_track = total_track.flatten()

            

            # can calculate sdr when we have estimate and reference
            if instance == instances[0]:
                # estimate     
                # write data in wav
                folder = f"{DATAFOLDER}predict"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                wavfile.write(f"{folder}/{self.song_name}.wav", SAMPLE_RATE, total_track)
            elif instance == instances[1]:
                # SDR is always 0.5 away from mus_eval SDR  thus it is correct to use on spectrogram too just mention how!
                # calculate song_sdr 
                self.song_sdr = np.mean(self.song_sdr)
                # calcualte song_mse
                self.song_mse = np.mean(self.song_mse) 
                self.calculate_loss(self.song_mse,self.song_sdr)
                 

                
                folder = f"{DATAFOLDER}target"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                wavfile.write(f"{folder}/{self.song_name}.wav", SAMPLE_RATE, total_track)
            else:
                folder = f"{DATAFOLDER}mixture"
                if not os.path.exists(folder):
                    os.makedirs(folder)
                wavfile.write(f"{folder}/{self.song_name}.wav", SAMPLE_RATE, total_track)

    def save_wave_form(self):
        pass

    def visualize_spectrogram(self):
        pass

    def calculate_loss(self,song_mse,song_sdr):
        print(f"song {self.song_name.split('-')[0]}\t{round(song_mse,5)}")
        print(f"song {self.song_name.split('-')[0]}\t{round(song_sdr,5)}")
        self.database_mse.append(song_mse)
        self.database_sdr.append(song_sdr)
        folder = f"{DATAFOLDER}"
        np.save(f"{folder}database_mse",np.array(self.database_mse))
        np.save(f"{folder}database_sdr",np.array(self.database_sdr))
        self.song_mse = []
        self.song_sdr = []
        self.estimate_list = []

    def _load_spectrograms(self, x_mixture_file_chunks, y_target_file_chunks):
        x_mixture = []
        y_target = []
        assert len(
            x_mixture) < 400, f"program does not support individual input tracks longer than 20 minutes"
        for i, file in enumerate(x_mixture_file_chunks):
            normalized_spectrogram = np.load(file)
            x_mixture.append(normalized_spectrogram)
        for i, file in enumerate(y_target_file_chunks):
            normalized_spectrogram = np.load(file)
            y_target.append(normalized_spectrogram)
        x_mixture = np.array(x_mixture)
        y_target = np.array(y_target)
        return x_mixture, y_target


def main():
    auto_encoder = AutoEncoder.load(
        "Final_Model_Other-10-0.01871-0.03438 VALID")
    # btrain = mixture, y_target = target
    x_mixture, y_target = load_fsdd("test")

    total_track = []
    estimate = None
    reference = None
    for r in range(3):
        # r=2
        total_track = []
        # test 140-160 should be very good! [8, 56, 112, 216, 312, 560]
        for chunk in range(180, 181, 1):
            if r == 0:
                # predict
                x_train = auto_encoder.model.predict(x_mixture[chunk:chunk+1])
                y_targ = y_target[chunk:chunk+1]
                mse = MeanSquaredError()
                mse = mse(x_train, y_targ).numpy()
                print("error\t\t", mse)
            if r == 1:
                # target
                # y_target when vocal source sep
                x_train = np.array(y_target[chunk:chunk+1])
            if r == 2:
                # mixture
                x_train = np.array(x_mixture[chunk:chunk+1])

            mute_sound = False
            if -0.15 < np.min(x_train) < 0.15 and -0.15 < np.max(x_train) < 0.15 and -0.15 < np.mean(x_train) < 0.15:
                print("mute chunk")
                mute_sound = True

            # min max normalizing
            min_max_normalizer = MinMaxNormalizer(0, 1)
            x_train = min_max_normalizer.denormalize(x_train)

            # fixing dimensions with input
            x_train = x_train[:, :, :, 0]
            x_train = x_train[0]
            # remove last timestep that was added with 0's
            x_train = x_train[:, :127]
            # add 1 more frequency bin that is same as last frequency
            x_train = np.vstack((x_train, x_train[-1]))
            x_train = librosa.db_to_amplitude(x_train)

            if VISUALIZE:
                amp_log_spectrogram = librosa.amplitude_to_db(
                    x_train, ref=np.max)
                fig, ax = plt.subplots()
                img = librosa.display.specshow(amp_log_spectrogram, y_axis='linear', sr=SAMPLE_RATE, hop_length=HOP_LENGTH,  x_axis='time', ax=ax)
                ax.set(title='Log-amplitude spectrogram')
                ax.label_outer()
                fig.colorbar(img, ax=ax, format="%+2.f dB")
                plt.show()

            # calculate estimate and reference spectrogram
            if r == 0:
                estimate = x_train
            if r == 1:
                reference = x_train

                # defossez calculation mean std
                # reference2 = (reference - np.mean(reference)) / np.std(reference)
                # estimate = (estimate - np.mean(reference)) / np.std(reference)
                # reference = reference2

            # generate waveform
            x_source = librosa.griffinlim(
                x_train, hop_length=HOP_LENGTH, n_fft=N_FFT)

            # fix waveform is silence
            if mute_sound:
                # +0.001 s.t no noise is heard
                x_source = np.zeros_like(x_source)+0.001

            # 132300 = 3 second excerpt
            assert x_source.shape[0] == SAMPLE_RATE * \
                CHUNK_DURATION, f"chunk is too short {x_source.shape}"
            total_track.append(x_source)

        total_track = np.array(total_track)
        total_track = total_track.flatten()

        # can calculate sdr when we have estimate and reference
        if r == 0:
            # estimate
            wavfile.write("mss_evaluate_data/other_predict.wav",
                          SAMPLE_RATE, total_track)
        elif r == 1:
            # reference
            delta = 1e-7  # avoid numerical errors
            print(reference.shape, estimate.shape)
            # 0 is column, 1 is row
            num = np.sum(np.square(reference), axis=(0, 1))
            den = np.sum(np.square(reference - estimate), axis=(0, 1))
            num += delta
            den += delta
            # SDR is always 0.5 away from mus_eval SDR  thus it is correct to use on spectrogram too just mention how!
            print("sdr:\t", 10 * np.log10(num / den))

            wavfile.write("mss_evaluate_data/other_target.wav",
                          SAMPLE_RATE, total_track)
        else:
            wavfile.write("mss_evaluate_data/other_mixture.wav",
                          SAMPLE_RATE, total_track)


if __name__ == "__main__":
    main()
