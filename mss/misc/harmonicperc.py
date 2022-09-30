from __future__ import print_function
import musdb

import tensorflow as tf
import pandas as pd
from scipy.io import wavfile

import numpy as np
import matplotlib.pyplot as plt

import librosa
import librosa.display
from matplotlib.pyplot import close
# this function splits the music tracks on alphabetical order instead of order in directory
# mus_train = musdb.DB(root="databases/database",subsets="train", split='train',download=False,is_wav=False)
# mus_valid = musdb.DB("database_wav",subsets="train", split='valid',download=False,is_wav=False)
mus_train = musdb.DB(root="databases/database",subsets="test",download=False,is_wav=False)
import os

# wav = load_track("database/train/Steven Clark - Bounty.stem.mp4", 2, 44100)
import numpy as np

VISUALISATION_FOLDER = "mss_evaluate_data/visualisation/"
# sdr_db = []
# sdr_list = []
def calculate_sdr(reference,chunk,T):
    delta = 1e-7  # avoid numerical errors
    # 0 is column, 1 is row
    num = np.sum(np.square(reference), axis=(0, 1))
    den = np.sum(np.square(reference - T), axis=(0, 1))
    num += delta
    den += delta
    sdr = 10 * np.log10(num / den)
    name = f"{chunk}\tsong-{chunk}-{chunk}\sdr\t"
    print("sdr\t",sdr)
    return sdr   
# make this a generator function
def iterate_tracks(tracks): # this functions stops when it yields the values 
    sdr_list = []
    sdr_db = []
    for i,track in enumerate(tracks):
        
        # print(track.audio.shape)
        # track.audio = np.zeros(10000)
        track.chunk_duration = 20.0
        max_chunks = int(track.duration/track.chunk_duration)
       
        # print(5/0)
        for j in range (0,max_chunks):
            track.chunk_start = j * track.chunk_duration 
            x = (track.audio) # don't transpose it
            y = (track.targets["other"].audio)
            # y+= (track.targets["vocals"].audio)
            y= y.transpose()
            if y.shape[0] <=2: y = y.transpose() 
            x= x.transpose()
            if x.shape[0] <=2: x = x.transpose() 
            
            try: 
                y,_  = y[:,0], y[:,1]
                x,_  = x[:,0], x[:,1]
            except:
                pass
           
            D = librosa.stft(x)
            T = librosa.stft(y)   

            D_harmonic, D_percussive = librosa.decompose.hpss(D)
            D_harmonic = np.abs(D_harmonic)
            T = np.abs(T)
            D_harmonic = librosa.amplitude_to_db(D_harmonic)
            T = librosa.amplitude_to_db(T)

            sdr_list.append(calculate_sdr(D_harmonic,j,T))

         


            # D = librosa.istft(D_harmonic)
            # print(D.shape)
            # wavfile.write(f"harmonic.wav",44100,D)
            
            # asd
            # print(x.shape)
            # # print(y.shape)
            # print(i)
            # break
            # Pre-compute a global reference power from the input spectrum
            # rp = np.max(np.abs(D))

            # plt.figure(figsize=(12, 8))
            # plt.subplot(3, 1, 1)
            # librosa.display.specshow(librosa.amplitude_to_db(D, ref=rp), y_axis='log')
            # plt.colorbar()
            # plt.title('Full spectrogram')            

            # plt.subplot(3, 1, 2)
            # librosa.display.specshow(librosa.amplitude_to_db(D_harmonic, ref=rp), y_axis='log')
            # plt.colorbar()
            # plt.title('Harmonic spectrogram')

            # plt.subplot(3, 1, 3)
            # librosa.display.specshow(librosa.amplitude_to_db(D_percussive, ref=rp), y_axis='log', x_axis='time')
            # plt.colorbar()
            # plt.title('Percussive spectrogram')
            # plt.tight_layout()

            # plt.show()
        '''
        if i <1:
            pass
            # if not os.path.exists("foldertest"):
            #     os.makedirs("foldertest")
            # wavfile.write(f"foldertest/mix_track-{i}-chunk-{j}.wav",44100,x)
            # wavfile.write(f"foldertest/target_v_track-{i}-chunk-{j}.wav",44100,y)
        ''' 
        # else:
        # print(sdr_list)
      
        sdr_db.append(np.mean(sdr_list))
        fig = plt.figure()
        ax = plt.subplot(111)
        ax.plot(sdr_db, label='sdr loss')
        ax.plot(np.ones_like(sdr_db)*np.mean(sdr_db), label = 'avg sdr')
        ax.set(title='SDR loss per song in MUSDB test')
        plt.xlabel("test song")
        plt.ylabel("SDR")
        ax.legend()
        folder = f"{VISUALISATION_FOLDER}loss_metrics/"
        if not os.path.exists(folder): os.makedirs(folder)
        plt.show()
        asd
        np.save(f"{folder}database_sdr_harmonic",np.array(sdr_db))
        fig.savefig(f"{folder}0_sdr_loss_harmonic")
        close(fig)
        print(np.mean(sdr_list))
        sdr_list = []
        # if i >1:
        #     print(5/0)


# iterate_tracks(mus_train)
print("finished")
# sdr_db = np.load(file=r"E:/Documenten E/University/Jaar 5/Project MSS/Workspace-MasterThesis-MSS/mss_evaluate_data/visualisation/loss_metrics/database_sdr_harmonic.npy") 

sdr_db_harm =           np.load(file=r"E:/Documenten E/University/Jaar 5/Project MSS/Workspace-MasterThesis-MSS/mss_evaluate_data/visualisation/loss_metrics/database_sdr_harmonic.npy") 
sdr_db_raw =            np.load(file=r"E:/Documenten E/University/Jaar 5/Project MSS/Workspace-MasterThesis-MSS/mss_evaluate_data/visualisation/loss_metrics/database_sdr_4.npy") 
sdr_db_postprocessed =  np.load(file=r"E:/Documenten E/University/Jaar 5/Project MSS/Workspace-MasterThesis-MSS/mss_evaluate_data/visualisation/loss_metrics/database_sdr_3.npy") 

import pandas as pd
import researchpy as rp
import scipy.stats as stats
my_array =  np.array([sdr_db_raw,sdr_db_postprocessed])

df = pd.DataFrame()
df["sdr_raw"] = sdr_db_raw
df["sdr_postprocessed"] = sdr_db_postprocessed
# df = pd.DataFrame(my_array, index=['sdr_harmonic','sdr_raw','sdr_postprocessed'])
# df.columns = np.arange(0,51)
summary,results = rp.ttest(df["sdr_raw"],df["sdr_postprocessed"])
print(summary)
print(results)

asd

# my_array =  np.array([sdr_db_harm,sdr_db_raw,sdr_db_postprocessed])
# df = pd.DataFrame(my_array, index=['sdr_harmonic','sdr_raw','sdr_postprocessed'])
# df.columns = np.arange(0,51)
# print(df.columns)
import scipy.stats as stats
from statsmodels.stats.multicomp import pairwise_tukeyhsd
# https://www.youtube.com/watch?v=qOw9jHYw89Y&ab_channel=VincentStevenson&loop=0

groups = [list(sdr_db_harm),list(sdr_db_raw),list(sdr_db_postprocessed)]

# if p val < 0.05 then there is a difference!
f_val, p_val = stats.f_oneway(*groups)
print(f_val,p_val)

data_1d = list(sdr_db_harm)+list(sdr_db_raw)+list(sdr_db_postprocessed)
groups_1d =[50*["harmonic/percusive"],50*["raw"],50*["postprocessed"]]
flatten = lambda x: [y for l in x for y in flatten(l)] if type(x) is list else [x]
groups_1d = flatten(groups_1d)
print(len(data_1d))
print(len(groups_1d))
tukey_test = pairwise_tukeyhsd(data_1d,groups_1d,alpha=0.05)
print(tukey_test.summary())

p_vals = [round(x,3) for x in tukey_test.pvalues]
print("harmonic\t\t","mean: ",np.round(np.mean(sdr_db_harm),2),"SD ",np.round(np.std(sdr_db_harm),2),"p_val: ",p_vals[0]) 
print("raw\t\t\t","mean: ",np.round(np.mean(sdr_db_raw),2),"SD ",np.round(np.std(sdr_db_raw),2),"p_val: ",p_vals[1])
print("postprocessed\t\t","mean: ",np.round(np.mean(sdr_db_postprocessed),2),"SD ",np.round(np.std(sdr_db_postprocessed),2),"p_val: ",p_vals[2])



asd

# asd
# print(df.summary)
import researchpy as rp






fig = plt.figure()
ax = plt.subplot(111)
ax.plot(sdr_db, label='sdr loss')
ax.plot(np.ones_like(sdr_db)*np.mean(sdr_db), label = 'avg sdr')
ax.set(title='SDR loss per song in MUSDB test')
plt.xlabel("test song")
plt.ylabel("SDR")
ax.legend()
plt.show()
asd
# folder = f"{VISUALISATION_FOLDER}loss_metrics/"
# if not os.path.exists(folder): os.makedirs(folder)

# np.save(f"{folder}database_sdr_harmonic",np.array(sdr_db))
# fig.savefig(f"{folder}0_sdr_loss_harmonic")
# close(fig)
# print(np.mean(sdr_list))
