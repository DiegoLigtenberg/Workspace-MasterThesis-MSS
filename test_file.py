
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
file = "original.wav"
'''able to convert any .wav file to spectrogram in pytorch and back''' 

import random
# torch.set_printoptions(precision=10)
#numpy array     
# 
#          
# 
# for i in range(10):
#     print(type(random.choice([-2,2])))
for i in range (0,3):
    print(i)

# aa
# signal,sr = librosa.load(file,mono=False,sr=44100,duration=3.0)
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
