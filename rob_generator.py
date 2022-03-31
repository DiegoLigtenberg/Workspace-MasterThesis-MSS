from audioop import minmax
from preprocesssing import MinMaxNormalizer

import numpy as np
import matplotlib.pyplot as plt
# from aa import VariationalAutoEncoder
from auto_ancoder_lolstaatdaarmssnietvoorxD import VariationalAutoEncoder
from atrain import load_fsdd
import librosa, librosa.display
from scipy.io import wavfile
from scipy.signal import wiener
if __name__=="__main__":
    variational_auto_encoder = VariationalAutoEncoder.load("model_skipcon")  #model_spectr for first_source_sep
    x_train,y_train = load_fsdd("train")
    x_train = variational_auto_encoder.model.predict(x_train[:5])    
    min_max_normalizer = MinMaxNormalizer(0,1)   
    x_train = min_max_normalizer.denormalize(x_train)
    x_train = x_train [:,:,:,0]  
    x_train = x_train[0]
    x_train = x_train[:,:127]
    x_train = x_train[:-1]   

    
    x_train = librosa.db_to_amplitude(x_train)     

    #visualize spectrogram
    amp_log_spectrogram = librosa.amplitude_to_db(x_train,ref=np.max)   
    fig, ax = plt.subplots()      
    img = librosa.display.specshow(amp_log_spectrogram, y_axis='linear', sr=44100, hop_length=1050,  x_axis='time', ax=ax)
    ax.set(title='Log-amplitude spectrogram')
    ax.label_outer()
    fig.colorbar(img, ax=ax, format="%+2.f dB")
    plt.show()

    x_source = librosa.griffinlim(x_train,hop_length=1050)
    wavfile.write("wtf2.wav",44100,x_source) 
