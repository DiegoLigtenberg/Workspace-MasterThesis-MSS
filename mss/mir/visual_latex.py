import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import close
import os 
import glob

import librosa, librosa.display
from mss.settings.settings import N_FFT,HOP_LENGTH,SAMPLE_RATE,CHUNK_DURATION,MONO

from mss.utils.dataloader import natural_keys

import musdb
from scipy.io import wavfile

import pandas as pd
# list of named collors in matplotlib
#https://matplotlib.org/stable/gallery/color/named_colors.html

# https://librosa.org/doc/0.7.2/generated/librosa.display.waveplot.html

input_path = r"E:/Documenten E/University/Jaar 5/Project MSS/Thesis Latex/Chapter 2 - Data/2.1 sound"
output_path = r"E:/Documenten E/University/Jaar 5/Project MSS/Thesis Latex/Chapter 2 - Data/2.1 sound"

generic_instruments = ["violin","guitar","voice"] # ["mixture","vocals","drums","bass","other","instruments"] 
# generic_instruments = ["mixture","vocals","drums","bass","other","instruments"] # ["violin","guitar","voice"]



dataset_output_path = r"E:/Documenten E/University/Jaar 5/Project MSS/Thesis Latex/Chapter 2 - Data/2.2 audio dataset/data distribution"
def save_dataset_distribution(output_path,traintest):
    data = [pd.read_csv("MIR_datasets/MIR_train_labels_merged.csv"),pd.read_csv("MIR_datasets/MIR_test_labels_combined.csv")]
    font_size = 22
    i = traintest
    df_train = data[i]
    ax = df_train.sum().plot(kind='bar', figsize=(14, 10), title='',
            xlabel='instrument',fontsize=font_size, ylabel='amount', legend=False)
    plt.xlabel("instrument",fontsize=font_size)
    plt.ylabel("amount",fontsize=font_size)
    ax.margins(y=0.12,x=0.1)
    for index,data in enumerate(df_train.sum()):
        if i == 0:
            plt.text(x=index , y = data+15 , s=f"{round(100*(data/sum(df_train.sum())),2)}%" , fontdict=dict(fontsize=font_size),ha="center")
        if i == 1:
            plt.text(x=index , y = data+3 , s=f"{round(100*(data/sum(df_train.sum())),2)}%" , fontdict=dict(fontsize=font_size),ha="center")


    if i == 0:
        plt.savefig(f"{output_path}/train.jpg");plt.close()
    if i == 1:
        plt.savefig(f"{output_path}/test.jpg");plt.close()


def save_waveform_img(input_path,output_path ):
    data = glob.glob(os.path.join(input_path, '*.wav'),recursive=True)            
    data.sort(key=natural_keys)    
    
    for i, file in enumerate(data):
        y,sr = librosa.load(file,sr=SAMPLE_RATE,duration=3)
        plt.figure()

        #waveform
        librosa.display.waveplot(y,sr=SAMPLE_RATE) # colors: color="red" https://matplotlib.org/stable/gallery/color/named_colors.html
        # plt.title(generic_instruments[i])
        plt.xlabel("time")
        plt.ylabel("amplitude")
        plt.savefig(f"{output_path}/waveform/{i}_{generic_instruments[i]}.jpg")
        plt.close()
        # fft
        ft = np.fft.fft(y)
        magnitude_spectrum = np.abs(ft)
        frequency = np.linspace(0,sr,len(magnitude_spectrum))
        num_frequency_bins = int(len(frequency) * 0.5) # 50% because it is symmetric
        plt.plot(frequency[:num_frequency_bins],magnitude_spectrum[:num_frequency_bins])
        plt.xlabel("frequency (hz)")
        plt.ylabel("amplitude")
        plt.savefig(f"{output_path}/fourier transform/{i}_{generic_instruments[i]}.jpg")
        plt.close()
        # spectrogram
        
        '''
        mag,phase = librosa.magphase(np.abs(librosa.stft(y)))
        mag = mag**2
        # print(D,"\n\BOOB\n")
        mag = (10*np.log10(mag))
        print(mag,"\n\BOOB\n")
        # asd
        S = np.abs(librosa.stft(y)).shape
        '''

        D = librosa.amplitude_to_db(np.abs(librosa.stft(y,n_fft=N_FFT,hop_length=HOP_LENGTH)))
        # asd
        print(D.shape)
        asd
        librosa.display.specshow(D, y_axis='linear')
        plt.colorbar(format='%+2.0f dB')
        plt.xlabel("time")
        # plt.title(f'{generic_instruments[i]}')
        # plt.savefig(f"{output_path}/spectrogram/{i}_{generic_instruments[i]}.jpg")
        print(file)
        # plt.close()
        plt.show()
        
# a = np.load(r"G:/Thesis/valid/mixture/0-0-0.npy")
# print(a.shape)
# asd
save_waveform_img(input_path,input_path)
asd

def gain_musdb_sounds():
    mus_train = musdb.DB(root="databases/database",subsets="test",download=False,is_wav=False) #is_wav = false because database_wav folder is in mp4
    for i,track in enumerate(mus_train):
        if i ==10 :
            print(track.audio.shape)
            # track.audio = np.zeros(10000)
            track.chunk_duration = 30.0
            max_chunks = int(track.duration/track.chunk_duration) 
        


            # print(5/0)
            for j in range (0,max_chunks):
                if j ==1:
                    track.chunk_start = j * track.chunk_duration 
                    x = (track.audio) [15*SAMPLE_RATE:25*SAMPLE_RATE] # don't transpose it
                    y = (track.targets["vocals"].audio[15*SAMPLE_RATE:25*SAMPLE_RATE])                
                    z = (track.targets["other"].audio[15*SAMPLE_RATE:25*SAMPLE_RATE])
                    m = z + y
                    wavfile.write(f"{output_path}/instruments {j}_chunk1_song{i}.wav",SAMPLE_RATE,m)
                    # print(5/0)
                    break
        else:
            if i >=10:
                pass

def get_sdr_per_song():
    data = r"E:\Documenten E\University\Jaar 5\Project MSS\Workspace-MasterThesis-MSS\mss_evaluate_data\visualisation\loss_metrics/database_sdr_4.npy".replace("\\","/")
    data = np.load(data)
    # print(data)
    print(data)

# save_waveform_img(input_path,input_path)


# save_dataset_distribution(dataset_output_path,0)
# save_dataset_distribution(dataset_output_path,1)
# asd