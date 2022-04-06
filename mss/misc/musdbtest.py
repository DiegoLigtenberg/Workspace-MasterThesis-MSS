# https://github.com/sigsep/sigsep-mus-oracle/blob/master/IBM.py

import musdb
# from tensor_to_wav import save_wav
# mus = musdb.DB("database_wav",download=False,is_wav=True)
# mus[0].audio

import tensorflow as tf
import pandas as pd
from scipy.io import wavfile

# this function splits the music tracks on alphabetical order instead of order in directory
mus_train = musdb.DB(root="databases/database",subsets="train", split='train',download=False,is_wav=False)
# mus_valid = musdb.DB("database_wav",subsets="train", split='valid',download=False,is_wav=True)
# mus_test = musdb.DB("database_wav",subsets="test",download=False,is_wav=True)
import os

# wav = load_track("database/train/Steven Clark - Bounty.stem.mp4", 2, 44100)
import numpy as np
# make this a generator function
def iterate_tracks(tracks): # this functions stops when it yields the values 
    for i,track in enumerate(tracks):
        
        print(track.audio.shape)
        # track.audio = np.zeros(10000)
        track.chunk_duration = 10.0
        max_chunks = int(track.duration/track.chunk_duration)
       
        # print(5/0)
        for j in range (0,max_chunks):
            track.chunk_start = j * track.chunk_duration 
            x = (track.audio) # don't transpose it
            y = (track.targets["vocals"].audio)
            print(x.shape)
            # print(y.shape)
            print(i)
            # break
            if j <3:
                if not os.path.exists("foldertest"):
                    os.makedirs("foldertest")
                wavfile.write(f"foldertest/mix_track-{i}-chunk-{j}.wav",44100,x)
                wavfile.write(f"foldertest/target_v_track-{i}-chunk-{j}.wav",44100,y)
            else:
                print(5/0)
            # yield x,y  with yield need to upgrade
    # return x,y
    return [5,5]
            


            # print(len(x))
            # print(track.name)
            # outputsignal = tf.audio.encode_wav(x,44100,"tensorsong.wav")
            
            # print("encoded")
            # break
        # break
        # print( "broken")

# train_dataset = tf.data.Dataset.from_generator(generator=iterate_tracks,output_types=(tf.float64, tf.uint8))
# train_dataset = train_dataset.batch(32)

gen = iterate_tracks(mus_train)
print(len(gen))
# im_list = []




# for i in range(0,2):
#     print("boe")
#     x,y = next(gen)
#     im_list.append((x,y))
# print(len(im_list))
# iterate_tracks(mus_train)
# print(mus_train.tracks)

# print(5/0)
# print(len(mus_train))
# print(len(mus_valid))
# print(len(mus_test))
# for track in mus_test:
    # print(track.name)
    # print(track.targets['vocals'])


import random
import time

# get random 5 second segments from a song up until you reached everything 
def get_segments():
    while True:
        track = random.choice(mus_train.tracks)
        track.chunk_duration = 5.0
        track.chunk_start = random.uniform(0, track.duration - track.chunk_duration)
        x = track.audio.T
        y = track.targets['vocals'].audio.T
     
        print(x.shape)

        yield x,y

# print(len(list(get_segments())))

