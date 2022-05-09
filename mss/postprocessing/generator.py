from audioop import minmax
from math import prod
from mss.preprocessing.preprocesssing import MinMaxNormalizer

import numpy as np
import matplotlib.pyplot as plt
# from auto_encoder_vanilla import VariationalAutoEncoder
# from mss.models.auto_encoder import AutoEncoder
from mss.models.auto_encoder_other import AutoEncoder
from mss.models.atrain import load_fsdd
import librosa, librosa.display
from scipy.io import wavfile
from scipy.signal import wiener
import tensorflow as tf
from tensorflow.keras import backend as K


def main():
    auto_encoder = AutoEncoder.load("Final_Model_Other_extra_songs-15-0.01687-0.03398 VALID")  #model_spectr for first_source_sep
    auto_encoder.summary()
    b_train,y_train = load_fsdd("test") # note the amnt of datapoints load_fssd loads -> check the function
    (np.min(b_train),np.max(b_train))
    

    total_track = []
    reall = False
    estimate = None
    reference = None
    for r in range (1):
        # r=2
        total_track = []
        for i in range(380,381,1): # test 140-160 should be very good! [8, 56, 112, 216, 312, 560]
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

            if r == 0:
                
                # sub(x_train)
                # x_train[(x_train<0.07) & (x_train >= 0.0)]  -=.101
                # x_train[:128][(x_train[:128]<0.1) & (x_train[:128] >= 0.0)]  *=.2

                # x_train[::][(x_train[::]<0.0) & (x_train[::] > -0.2)]  /=.2 # the lower the division number (closer to 0) -> the more sound (drums) are removed, but also other sound
                # x_train[::][(x_train[::]<=-0.2)] = -.33

                # x_train[(x_train<=0.1)] = -.33
                x_train[:128][(x_train[:128]<0.1) & (x_train[:128] >= 0.0)]  *=.2 # the lower the multiplication number ( closer to 0) -> the more sounds are removed -> BAD actually EQ
                x_train[975::][(x_train[975::]<0.1) & (x_train[975::] >= 0.0)]  *=.1 # high rfequency's we hear  harder even when decrease is softer
                x_train[(x_train<0.0) & (x_train > -0.2)]  /=.2 # the lower the division number (closer to 0) -> the more sound (drums) are removed, but also other sound
                x_train[(x_train<=-0.2)] = -.33
                pass

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
            print(x_train.shape)
            
            # amp_log_spectrogram = librosa.amplitude_to_db(x_train,ref=np.max)
       
            # fig, ax = plt.subplots()      
            # img = librosa.display.specshow(amp_log_spectrogram, y_axis='linear', sr=44100, hop_length=1050,  x_axis='time', ax=ax)
            # ax.set(title='Log-amplitude spectrogram')
            # ax.label_outer()
            # fig.colorbar(img, ax=ax, format="%+2.f dB")
            # plt.show()

          
            if r == 0:
                estimate = x_train
            if r == 1:
                reference = x_train

                # reference2 = (reference - np.mean(reference)) / np.std(reference)
                # estimate = (estimate - np.mean(reference)) / np.std(reference)
                # reference = reference2
                
         
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
            # total_track = wiener(total_track,mysize=3)
            wavfile.write("track_output/other_predict.wav",44100,total_track) 
            # estimate = total_track
        elif r == 1:  
            # reference  = total_track          
            delta = 1e-7  # avoid numerical errors
            print(reference.shape)
            num = np.sum(np.square(reference), axis=None)
            # print(reference,estimate)
            den = np.sum(np.square(reference - estimate), axis=None)

            print(np.min(reference),np.max(reference),np.mean(reference))
            print(np.min(estimate),np.max(estimate),np.mean(estimate))
            print(reference.shape)
            print(reference[0])
            print(estimate[0])
       
    
            num += delta
            den += delta
            print(num)
            print(den,"\n")
            print("sdr:\t", 10 * np.log10(num  / den)) # SDR is always 0.5 away from mus_eval SDR  thus it is correct to use on spectrogram too just mention how! 
            # import museval
            # estimate = np.atleast_2d(estimate)
            # reference = np.atleast_2d(reference)
            # print(estimate.shape)
            # print(reference.shape)
            # print(museval.evaluate(estimates=estimate,references=reference))
        
            wavfile.write("track_output/other_target.wav",44100,total_track) 
        else:
            wavfile.write("track_output/other_mixture.wav",44100,total_track) 


    # print(x_train.shape)


if __name__=="__main__":
    main()