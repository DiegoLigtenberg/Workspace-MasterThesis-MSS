
from tensorflow.keras.utils import Progbar
import time 
import numpy as np

metrics_names = ['acc','pr'] 

num_epochs = 5
num_training_samples = 100
batch_size = 10

import tensorflow as tf
# import keras as keras
# a = tf.keras.initializers.TruncatedNormal(
#     mean=0.0, stddev=0.05, seed=None
# )
# print(a)


import random

a = ['a', 'b', 'c']
b = [0,1, 2, 3,4,5,6,7]
print(b[0:1])

c = list(zip(a, b))

random.shuffle(c)

a, b = zip(*c)

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