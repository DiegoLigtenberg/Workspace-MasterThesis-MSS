from audioop import minmax
from mss.preprocessing.preprocesssing import MinMaxNormalizer

import numpy as np
import matplotlib.pyplot as plt
# from auto_encoder_vanilla import VariationalAutoEncoder
from mss.models.auto_encoder import AutoEncoder
from mss.models.atrain import load_fsdd
import librosa, librosa.display
from scipy.io import wavfile
from scipy.signal import wiener

def main():
    auto_encoder = AutoEncoder.load("modelpostfix2")  #model_spectr for first_source_sep
    auto_encoder.summary()
    b_train,y_train = load_fsdd("train") # note the amnt of datapoints load_fssd loads -> check the function
    (np.min(b_train),np.max(b_train))
    x_train=np.array(b_train[3:4])
    # x_train += (np.random.rand(b_train.shape[0],b_train.shape[1],b_train.shape[2],b_train.shape[3])-0.5) * 0.3
    # print(x_train.shape)
    x_train = auto_encoder.model.predict(b_train[3:4])
    print(np.mean(np.abs((x_train-b_train[:1])**2)))
    print(np.min(x_train),np.max(x_train))
    error = (x_train-b_train[:1])*5
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
    
    # amp_log_spectrogram = librosa.amplitude_to_db(x_train,ref=np.max)
    # fig, ax = plt.subplots()      
    # img = librosa.display.specshow(amp_log_spectrogram, y_axis='linear', sr=44100, hop_length=1050,  x_axis='time', ax=ax)
    # ax.set(title='Log-amplitude spectrogram')
    # ax.label_outer()
    # fig.colorbar(img, ax=ax, format="%+2.f dB")
    # plt.show()

    # x_train = librosa.db_to_amplitude(x_train) 
    # x_source = wiener(x_train, (5, 5))
    # print(x_source.shape)
    x_source = librosa.griffinlim(x_train,hop_length=1050)
  
 
    # print(x_source)
    print("\n\n\n")
    # print(x_train.shape)
    # print(x_source.shape)
    wavfile.write("example_test.wav",44100,x_source) 

    # print(x_train.shape)


if __name__=="__main__":
    main()