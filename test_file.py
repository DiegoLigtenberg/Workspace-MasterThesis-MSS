
# from tensorflow.keras.utils import Progbar
import time 
import numpy as np
import librosa
metrics_names = ['acc','pr'] 

num_epochs = 5
num_training_samples = 100
batch_size = 10

import librosa, librosa.display
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import wavfile
    # import torch
file = r"track_output/"
'''able to convert any .wav file to spectrogram in pytorch and back''' 
from dora.log import LogProgress
import random
import logging
logger = logging.getLogger(__name__)
print(logger)
# torch.set_printoptions(precision=10)
#numpy array     
# 
#          
# 
# for i in range(10):
#     print(type(random.choice([-2,2])))

# aa
# a = np.array((  [1,2,3],
#                 [1,2,3],
#                 [1,2,3],
#                 [1,2,5]))

# b = np.array((  [2,2,3],
#                 [1,2,3],
#                 [1,2,5],
#                 [1,2,6]))
# print(a.shape)

# a = [1,2,3]
# # print(a == a[4])
# a = "123"
# b = f"{a}{None}"
# print(b)
# asd

import librosa
from pydub import AudioSegment
from scipy.io import wavfile
a = []

a = "OnlyMP3.net - Legends Never Die (ft. Against The Current) [OFFICIAL AUDIO]  Worlds 2017 - League of Legends-4Q46xYqUwZQ-192k-1638883605927.mp3"

# target_frequency = 6000 
# matrix_value = int((target_frequency/20000) *2048)
# print(matrix_value)
# matrix
a = [1,2,3,4]
print(len(a))
for i in a:
    print(i)
print(a.split(".")[-1])
base = ".".join((a.split(".")[:-1]))+"."
extension = "wav"
print(extension)
print(base+extension)
# if a.split(".")[1] != ".wav":
    # a = a.split(".")[-1:]+".wav"
# print(a)

asd
# sound = AudioSegment.from_file("track_output/other_predict.wav",format="wav")
raw,sound = wavfile.read('track_output/other_predict.wav')
shifted = sound * (2 ** 31 - 1)  
ints = shifted.astype(np.int32)
sound = AudioSegment(data=ints, sample_width=4, frame_rate=44100, channels=1)
sound.export("track_output/other_predict.mp3",format="mp3")
asd

import matplotlib.pyplot as plt
x_train = np.load("mss_evaluate_data/database_test/database_mse_original.npy")
y_train = np.load("mss_evaluate_data/database_test/database_sdr_original.npy")

print(x_train)
print(y_train,np.mean(y_train))
asd
fig = plt.figure()
ax = plt.subplot(111)
ax.plot(x_train, label='mse loss')
ax.plot(np.ones_like(x_train)* np.mean(x_train), label = 'avg mse')
ax.legend()
fig.savefig(f"visualisation/test_evaluation/mse_loss")


fig = plt.figure()
ax = plt.subplot(111)
ax.plot(y_train, label='sdr loss')
ax.plot(np.ones_like(y_train)*np.mean(y_train), label = 'avg sdr')
ax.legend()
fig.savefig(f"visualisation/test_evaluation/sdr_loss")

# a = [1,2,3,4,5,6,7,8]
# b = len(a)
# it = -1
# def mygen():
#     global it
#     it+=1
#     while it < b:
#         yield a[it]
    

# for i in range(b):
#     print(next(mygen()))

#  for i,chunk in enumerate(range(180,181,1)):
#      print(i)

# num = np.sum(np.square(a), axis=(0, 1))
            # print(reference,estimate)
# den = np.sum(np.square(a - b), axis=(0, 1))
# print(np.sum(np.square(a-b),axis=(0,1))) #0 = column #  1 = row 
'''
indexes = range(0, 14, 1)
indexes = LogProgress(logger, indexes, updates=1,
                        name='Eval')
for index in indexes:
    print(index)
    track = test_set.tracks[index]
'''
# signal,sr = librosa.load(file,mono=False,sr=44100)
# D = librosa.stft(signal)
# D_harmonic, D_percussive = librosa.decompose.hpss(D)

# source = librosa.griffinlim(D_harmonic,hop_length=1024)
# wavfile.write("track_output/librosagod.wav",44100,source) 

# print(signal.shape)
# signal_l, signal_r = signal[0], signal[1]
# print(signal_l.shape,signal_r.shape)
# signal = np.vstack((signal_l,signal_r))
# print(signal.shape)
# aa

# signal = np.mean(signal,axis=0)  
# augmented_signal = librosa.effects.pitch_shift(signal,44100,2.5)
# # augmented_signal *=0
# print(augmented_signal.shape)
# wavfile.write("original_strecth.wav",44100,augmented_signal) 
"""
stretch_factor = 0.5; stretch_factor = np.clip(stretch_factor,0.5,1)
print(5/stretch_factor)
aa
signal,sr = librosa.load(file,mono=False,sr=44100,duration=3.0)
signal = np.mean(signal,axis=0)
prev_shape  = signal.shape[0]
augmented_signal = librosa.effects.time_stretch(signal,0.5)
new_shape = augmented_signal.shape[0]
shape_range = new_shape - prev_shape
rng = random.randint(0,shape_range-1)
augmented_signal = augmented_signal[rng:rng+prev_shape:]
# print(augmented_signal.shape)
wavfile.write("original_strecth.wav",44100,augmented_signal) 
""" 



# tf.compat.v1.disable_eager_execution() 
# import random
# from tensorflow.keras.losses import MeanSquaredError
# y_true = np.array([[1., 1.], [1., 1.]])
# y_pred = np.array([[1., 1.], [0., 0.]])

# mse = MeanSquaredError()
# mse = mse(y_true,y_pred,sample_weight = np.array([1,0]))
# mse = mse.eval(session=tf.compat.v1.Session())
# print(mse)
# print(5/0)
# c = list(zip(a, b))

# random.shuffle(c)

# a, b = zip(*c)

# print (a)
# print (b)

# counter = 0
# def myfunc():
#         for i in range(num_epochs):
#                 global counter
              
#                 print("\nepoch {}/{}".format(i+1,num_epochs))
                
#                 pb_i = Progbar(num_training_samples, stateful_metrics=metrics_names)
                
#                 for j in range(num_training_samples//batch_size):
#                         counter +=1
                        
#                         time.sleep(0.3)
                        
#                         values=[('acc',np.random.random(1)), ('pr',counter)]
                        
#                         pb_i.add(batch_size, values=values)
                
# # myfunc()

# a = 5e-1 
# for i in range(5):
#         a/=2
#         print(a)
