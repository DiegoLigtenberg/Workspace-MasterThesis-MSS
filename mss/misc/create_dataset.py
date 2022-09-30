# https://github.com/sigsep/sigsep-mus-oracle/blob/master/IBM.py

import musdb
# from tensor_to_wav import save_wav
# mus = musdb.DB("database_wav",download=False,is_wav=True)
# mus[0].audio

import tensorflow as tf
import pandas as pd
from scipy.io import wavfile
import os 

# this function splits the music tracks on alphabetical order instead of order in directory
mus_train = musdb.DB(root="databases/database",subsets="train", split='train',download=False,is_wav=False) #is_wav = false because database_wav folder is in mp4
mus_valid = musdb.DB(root="databases/database",subsets="train", split='valid',download=False,is_wav=False)
print(len(mus_train))
print(len(mus_valid))
asd
# mus_test = musdb.DB("database_wav",subsets="test",download=False,is_wav=True)

# wav = load_track("database/train/Steven Clark - Bounty.stem.mp4", 2, 44100)
PATH = "database_chunk/train"


# make this a generator function
def create_dataset(tracks,folder): # this functions stops when it yields the values 
    for i,track in enumerate(tracks):
        track.chunk_duration = 5
        max_chunks = int(track.duration/track.chunk_duration)
        if not os.path.exists(folder+f"/{i}"):
                os.makedirs(folder+f"/{i}")
        for j in range (0,max_chunks):
            track.chunk_start = j * track.chunk_duration 
            x = (track.audio) # don't transpose it
            y1 = (track.targets["vocals"].audio)
            y2 = (track.targets["drums"].audio)
            y3 = (track.targets["bass"].audio)
            y4 = (track.targets["other"].audio)

            # print(x.shape)
            # print(y.shape)
            print(i,j,end="\r")
            if i <1:
                wavfile.write(f"{folder}/{i}/mixture-chunk-{j}.wav",44100,x)
                wavfile.write(f"{folder}/{i}/vocals-chunk-{i}-chunk-{j}.wav",44100,y1)                
                wavfile.write(f"{folder}/{i}/drum-chunk-{i}-chunk-{j}.wav",44100,y2)                
                wavfile.write(f"{folder}/{i}/bass-chunk-{i}-chunk-{j}.wav",44100,y3)                
                wavfile.write(f"{folder}/{i}/other-chunk-{i}-chunk-{j}.wav",44100,y4)

                # save_wav(f"track-{i}-chunk-{j}.wav",y)
            # yield x,y  with yield need to upgrade
    return x,y1,y2,y3,y4
            
create_dataset(mus_train,PATH)