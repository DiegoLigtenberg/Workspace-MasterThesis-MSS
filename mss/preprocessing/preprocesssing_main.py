''' 
1- load a file
2- extract segments
3- augment
4- pad the signal (if necessary)
5- extracting log spectrogram from signal
6- normalize spectrogram
7- save the normalized spectrogram

Preprocessing Pipeline
'''

# https://www.youtube.com/watch?v=O04v3cgHNeM&ab_channel=ValerioVelardo-TheSoundofAI&loop=0

from re import L
import librosa,librosa.display
import numpy as np
import os
import glob
# from view_spectrogram import spectro
import musdb
import math
import matplotlib.pyplot as plt
import pickle
import random
from pathlib import Path
from mss.utils.dataloader import natural_keys, atof
from mss.settings.settings import N_FFT,HOP_LENGTH,SAMPLE_RATE,CHUNK_DURATION,MONO

#TODO
# MAKE MAX CHUNKS + 1  because currently it ignroes last part of song

DATASET_DIR = "databases/database"
SPECTROGRAM_SAVE_DIR = "train_spectrogram"
MIN_MAX_VALUES_SAVE_DIR = "F:/Thesis/test"
TEMP_INFERENCE_SAVE_DIR = "temp_inference"       # do not change

AUGMENT = True

class Loader():
    '''Load is responsible for loading an audio file.'''
    def __init__(self,sample_rate,mono):
        self.sample_rate = sample_rate
        self.mono = mono
      
        self.input_track_list = glob.glob(os.path.join("track_input", '*/*.wav'),recursive=True)
        self.input_track_list.extend(glob.glob(os.path.join("track_input", '*/*.wav'),recursive=True))
        self.input_track_list.sort(key=natural_keys)
        self.input_track_len = len(self.input_track_list)  
        self.input_counter = 0    
        assert self.input_track_len > 0, f"there are should be .wav files in input directory:\tthere were {self.input_track_len} files found"  
        
    
    def load_from_path(self):
        '''this method can be implemented when I choose to make custom datasets'''     
        file = self.input_track_list[self.input_counter]
        signal = librosa.load(file,sr=self.sample_rate,mono=False)[0].T # transpose so stereo format is in (samples x stereo)        
        self.input_counter+=1
        return signal,file # returns a list of librosa loaded track waveforms

    def reset_input_counter(self):
        self.input_counter = 0

    def load_musdb(self): # this functions stops when it yields the values 
   
        self.mus_train = musdb.DB(root="databases/database",subsets="train", split='train',download=False,is_wav=False) #is_wav = false because database_wav folder is in mp4
        self.mus_valid = musdb.DB(root="databases/database",subsets="train", split='valid',download=False,is_wav=False)
        self.mus_test = musdb.DB(root="databases/database",subsets="test",download=False,is_wav=False)
        return self.mus_train,self.mus_valid,self.mus_test

class Augmentation():
    '''this class is responsible for performing certain data augmentation techniques'''
    def __init__(self,augment) -> None:
        self.amnt_augments = 1
        self.augmentations = [self.same,self.polarity_inversion,self.reverse_track,self.time_stretch,self.pitch_scaling]
        self.augment_rng = None
        
        # rolling stretch factor
        self.samples = int(SAMPLE_RATE * CHUNK_DURATION)
        self.stretch_factor = 0.5; self.stretch_factor = np.clip(self.stretch_factor,0.5,1)      
        prev_shape = self.samples
        new_shape = self.samples/self.stretch_factor
        self.shape_range = new_shape-prev_shape
        if augment:            
            self.amnt_augments = len(self.augmentations)

    def augment(self,signal,aug):  
        self.signal=signal     
        # tranpose librosa stuff -> make sure data format = [sample x channel]
        self.signal = self.signal.transpose()
        if self.signal.shape[0] <=2: self.signal = self.signal.transpose() 

        
        try: self.signal_l, self.signal_r = signal[:,0], signal[:,1]
        except IndexError: self.signal_l,self.signal_r = signal,signal # signal already in mono (this fixes for whole pipeline)           

        augmented_track_l = self.augmentations[aug](self.signal_l)   
        augmented_track_r = self.augmentations[aug](self.signal_r)    
        augmented_track = np.vstack((augmented_track_l,augmented_track_r)).transpose()  
        return augmented_track

    def same(self,signal):
        return signal

    def polarity_inversion(self,signal):
        polarity_inversed_signal = signal * -1
        return polarity_inversed_signal
    
    def reverse_track(self,signal):
        reversed_track = np.flip(signal)
        return reversed_track

    def time_stretch(self,signal):
        augmented_signal = librosa.effects.time_stretch(signal,self.stretch_factor)        
        augmented_signal = augmented_signal[self.rng:self.rng+self.samples:]
        return augmented_signal

    def pitch_scaling(self,signal):        
        pitched_signal = librosa.effects.pitch_shift(signal,self.samples,self.semni_tones)
        return pitched_signal

    def roll_rng(self):
        # rolls rng number that decides which slice of original track is taken in slowed down version
        self.rng = random.randint(0,self.shape_range-1)

        # rolls rng number of semni tones to scale up or down pitch scaling 
        self.semni_tones = random.choice([-4,-3,-2,2,3,4])
    
        
  
class Padder:
    '''responsible to apply zero padding to an array - works for stereo'''
    # input (x,channels) -> output (x+padded,channels)
    def __init__(self,mode="constant"):
        self.mode = mode
    
    def left_pad(self,array,num_missing_items):
        array_l,array_r = array[:,0], array[:,1]
        padded_array_l = np.pad(array_l,((num_missing_items),0),mode=self.mode)
        padded_array_r = np.pad(array_r,((num_missing_items),0),mode=self.mode)
        padded_array = np.vstack((padded_array_l,padded_array_r)).transpose()
        return padded_array

    def right_pad(self,array,num_missing_items):
        # pads array with 0's after the original array 
        array_l,array_r = array[:,0], array[:,1]
        padded_array_l = np.pad(array_l,(0,(num_missing_items)),mode=self.mode,constant_values=0)
        padded_array_r = np.pad(array_r,(0,(num_missing_items)),mode=self.mode,constant_values=0)
        padded_array = np.vstack((padded_array_l,padded_array_r)).transpose()
        return padded_array

class LogSpectroGramExtractor():
    '''extracts logspectrogram (in dB) from a time-series (audio) signal'''    
    
    def __init__(self,n_fft):
        self.n_fft = n_fft
        self.has_hop = False # makes sure that we only calculate hop length once        

    def extract_stft(self,signal):
        # in 2049,127
        # out 2048,128
        self.signal = signal
        self.signal = self.signal.transpose()
        if self.signal.shape[0] <=2: self.signal = self.signal.transpose() # librosa loads in [channel=2,sample] -> reverse it to get [sample, channel=2]
        if not self.has_hop:
            self.hop_length = self.get_hop_length(max(self.signal.shape),self.n_fft)
            self.has_hop = True

        if MONO:
            self.signal =  np.mean(self.signal, axis=1)
            stft = librosa.stft(self.signal,n_fft=self.n_fft,hop_length=self.hop_length)[:-1] #dimensions = (1+ (frame_size/2)  , num_frames)  1024 -> 513 -> 512 ([:-1])
            spectrogram = np.abs(stft)
            if np.mean(spectrogram) == 0:
                # add extra column for 128 power of 2
                spectrogram = np.append(spectrogram,np.array(2048*[[0,]]),axis=1)
                spectrogram = spectrogram[...,np.newaxis]
                return spectrogram
            log_spectrogram = librosa.amplitude_to_db(spectrogram)  #can only use amplitude_to_db,ref=np.max to get nice plot scale db_max = 0                
            # self.plot_spectrogram(spectrogram)
            
            log_spectrogram = np.append(log_spectrogram,np.array(2048*[[0,]]),axis=1)
            # revert by    norm_stft_l = norm_stft_l[:,:127] # goes from 128 to 127 to keep perfect 3 second spectrograms!
            log_spectrogram = log_spectrogram[...,np.newaxis] # add new axis to match input with stereo  
            # print(log_spectrogram.shape)
            return log_spectrogram
        else:           
            signal_l, signal_r = self.signal[:,0], self.signal[:,1]  # take self.signal[0] if it's loaded from librosa  
            stft_l = librosa.core.stft(signal_l,hop_length=self.hop_length,n_fft=self.n_fft,center=True)[:-1] # remove highest frequency does not matter
            stft_r = librosa.core.stft(signal_r,hop_length=self.hop_length,n_fft=self.n_fft,center=True)[:-1] 
            spectrogram_l = np.abs(stft_l)
            spectrogram_r = np.abs(stft_r)
            spectrogram_l = np.append(spectrogram_l,np.array(2048*[[0,]]),axis=1)
            spectrogram_r = np.append(spectrogram_r,np.array(2048*[[0,]]),axis=1)
            if np.mean(stft_l) == 0 and np.mean(stft_r) ==0:
                spectrogram = np.dstack((spectrogram_l,spectrogram_r))
                return spectrogram
            log_spectrogram_l = librosa.amplitude_to_db(spectrogram_l)
            log_spectrogram_r = librosa.amplitude_to_db(spectrogram_r)
            log_spectrogram_stereo = np.dstack((log_spectrogram_l,log_spectrogram_r))
            return log_spectrogram_stereo  

    def plot_spectrogram(self,spectrogram):
        amp_log_spectrogram = librosa.amplitude_to_db(spectrogram,ref=np.max)
        fig, ax = plt.subplots()      
        img = librosa.display.specshow(amp_log_spectrogram, y_axis='linear', sr=SAMPLE_RATE, hop_length=self.hop_length,
                         x_axis='time', ax=ax)
        ax.set(title='Log-amplitude spectrogram')
        ax.label_outer()
        fig.colorbar(img, ax=ax, format="%+2.f dB")
        print(self.hop_length,self.n_fft)
        plt.show()
        print(5/0)

    def get_hop_length(self,amnt_samples,n_fft):
        min_hop_size = max(n_fft/5,700)
        max_hop_size = min(n_fft/3.5,1500)
        divs = [1]
        for i in range(2,int(math.sqrt(amnt_samples))+1):
            if amnt_samples%i == 0: divs.extend([i,amnt_samples/i])
        divs.extend([amnt_samples])
        divs = [x for x in divs if x > min_hop_size and x < max_hop_size]
        if not list(divs): divs = [int(n_fft/4)]
        return int(sorted(list(set(divs)))[-1])

 

class MinMaxNormalizer:
    #TODO WANT A GLOBAL MIN/MAX NORMALISER  -> FOUND GLOBAL MIN of STFT ~ -100    GLOBAL MAX ~+50 -> values bound between -0.66 and 0.33 (range 1)
    '''
    MinMaxNormalizer applies min max normalisation to an array 
    parmeters 
    -> min: minumum normalized value
    -> max: maximum normalized value
    '''
    def __init__(self,min_val,max_val):
        self.min = min_val
        self.max = max_val
        self.empty_segments = 0
        

    def normalize(self,array):
        # normalise has problems when array is padded -> fix later for last seconds of song
        if array.max() - array.min()==0:
            self.empty_segments+=1     
            return array
        norm_array = (array/150)
        # norm_array = (array - array.min()) / (array.max()-array.min())
        # norm_array = norm_array * (self.max - self.min) + self.min
        return norm_array

    def denormalize(self,norm_array): #,original_min,original_max):
        # array = (norm_array - self.min) / (self.max - self.min)
        # array = array * (original_max - original_min) + original_max
        array= (norm_array*150)
        return array

class Saver:
    def __init__(self,min_max_values_save_dir):                   
        self.min_max_values_save_dir = min_max_values_save_dir
    # np_array - source - train/val/test - i,j (track/sgement)
    def save_feature(self,feature,source,dataset_type,i,j,aug,inference=False):     # a = augment number  
        if inference == False:         
            save_dir = f"{dataset_type}/{source}"
            save_dir = Path("G:/Thesis")/Path(save_dir)
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            save_file = f"{dataset_type}/{source}/inference-{i}-{j}-{aug}"   
            save_file = Path("G:/Thesis")/Path(save_file)   
            np.save(str(save_file)+".npy",feature)
        else:
            save_dir = TEMP_INFERENCE_SAVE_DIR
            if not os.path.exists(save_dir): os.makedirs(save_dir)
            save_file = f"inference-{i}-{j}-{aug}"   
            save_file = Path(save_dir)/Path(save_file)   
            np.save(str(save_file)+".npy",feature)
        return save_file
        
    def save_min_max_values(self,min_max_values):
        save_path = os.path.join(self.min_max_values_save_dir,"min_max_values.pkl")
        # self._save(min_max_values,save_path) # put on later
    
    @staticmethod
    def _save(data,save_path):
        with open (save_path,"wb") as f:
            pickle.dump(data,f)





class PreprocessingPipeline:
    '''
    PrprocesspingPipeline processes audio file in a directory. applying the following steps
    1- load a file
    2- extract segments
    3- augment
    4- pad the signal (if necessary)
    5- extracting log spectrogram from signal
    6- normalize spectrogram
    7- save the normalized spectrogram
    '''

    def __init__(self,chunk_duration):
        self._loader = None
        self.chunk_duration = chunk_duration
        self._num_expected_samples = None

        self.augmentor = None
        self.padder = None
        self.extractor = None
        self.normalizer = None
        self.min_max_values = {}

        self.saver = None

        self.minimum_val = np.Inf 
        self.maximum_val = -np.Inf

        self.proces_iterator = -1

    @property
    def loader(self):
        return self._loader
    
    @loader.setter
    def loader(self,loader):
        self._loader = loader
        self._num_expected_samples = int(self._loader.sample_rate * self.chunk_duration)

    def process_path(self,audio_files_dir):
        for root, _, files in os.walk(audio_files_dir):
            for file in files:
                file_path = os.path.join(root, file)
                self._process_file(file_path)

    
    def process_database(self):
        self.dataset_type = "train"
       
        for i in range(0,3): # for train, val, test data\
            # self.dataset_type = "test"
            if i == 1: 
                self.dataset_type = "valid" 
                # make sure that valid and test don't use augments
                self.augmentor.amnt_augments = 1 
                self.saver.min_max_values_save_dir = self.dataset_type
                
            
            if i == 2: 
                self.dataset_type = "test"
                self.saver.min_max_values_save_dir = self.dataset_type
            
            #   format:    [track-chunk-augmentation]
            for j,track in enumerate(self.loader.load_musdb()[i]): #load_mus_db has train valid test 
                # augment data here
                # CANNOT AUGMENT BECAUSE track.audio file will not allow us to iterate over multiple datapoints 
                
                track.chunk_duration = self.chunk_duration
                max_chunks = int(track.duration/track.chunk_duration) +1 # +1 captures last few seconds of song < chunk duration -> needs padding   

                for k in range (0,max_chunks):   
                    track.chunk_start = k * track.chunk_duration           
                    mixture = (track.audio) # don't transpose it
                    vocal_target = (track.targets["vocals"].audio)
                    # bass_target = (track.targets["bass"].audio)
                    # drums_target = (track.targets["drums"].audio)
                    other_target = (track.targets["other"].audio)
                    # accompaniment = bass + drums + other
                    # accompaniment_target = (track.targets["accompaniment"].audio)
                    '''ADD VOCALS BECAUSE WE ALSO RECOGNIZE THEM IN THESIS'''
                    other_target += vocal_target 

                    multi_track_keys = ["mixture","vocals","other"]
                    multi_track_values = [mixture,vocal_target,other_target]      
                    multi_tracks = dict(zip(multi_track_keys,multi_track_values))          

                    # proces the audio segments
                    self._process_file(multi_tracks,j,k)
                    # print("min: ",self.minimum_val," max: ",self.maximum_val) #get min and max val for stft
                    # print(5/0)
                    
            self.saver.save_min_max_values(self.min_max_values) # should be outside loop
            print(f"empty segments in all {self.dataset_type}: {self.normalizer.empty_segments}")
            self.normalizer.empty_segments = 0
        print("finished.")
                
    def process_input_track(self):
        self.dataset_type = "inference"
        self.augmentor.amnt_augments = 1 # removes augmentation
        for j in range(self.loader.input_track_len):
            full_mixture = self.loader.load_from_path()[0]
            chunk = self._num_expected_samples
            max_chunks = math.ceil(full_mixture.shape[0] / chunk)
            for k in range(0,max_chunks):
                mixture = full_mixture[k*chunk:(k+1)*chunk]
                multi_track_keys = ["mixture"]
                multi_track_values = [mixture]      
                multi_tracks = dict(zip(multi_track_keys,multi_track_values)) 
                self._process_file(multi_tracks,j,k)
    
    def proces_input_track_generator(self):
        self.dataset_type = "inference"
        self.augmentor.amnt_augments = 1 # removes augmentation
        
       
        while self.proces_iterator < self.loader.input_track_len:
            self.proces_iterator+=1            
            j = self.proces_iterator
            full_mixture, file_name = self.loader.load_from_path()
            file_name = file_name.split("track_input")[1]
            chunk = self._num_expected_samples
            max_chunks = max(1,math.ceil(full_mixture.shape[0] / chunk)) # if < 3 seconds, i.e. not padded yet
            for k in range(0,max_chunks):
                mixture = full_mixture[k*chunk:(k+1)*chunk]
                multi_track_keys = ["mixture"]
                multi_track_values = [mixture]      
                multi_tracks = dict(zip(multi_track_keys,multi_track_values)) 
                self._process_file(multi_tracks,j,k,inference=True)
            yield file_name

    def _process_file(self,multi_tracks,j,k,inference=False):
        # roll_rng fixes rng rol for time stretching (preserves sound duration of 3 seconds) and semni tones for pitch scaling
        self.augmentor.roll_rng() 

        # [chunk-augment]
        for source in multi_tracks.keys():                     
            # signal = multi_tracks[source] # if it is here, re_use previosu signal augmentation (stacks augmentations) every time ( not what we want )
            for aug in range (0,self.augmentor.amnt_augments):
                # augment each source chunk
                signal = multi_tracks[source]
                
                signal = self.augmentor.augment(signal,aug)

                # pad each source chunk
                if self._is_padding_neccessary(signal):
                    signal = self._apply_padding(signal)
                    # print("padded to:\t",signal.shape)
                # print("mean",np.mean(np.mean((signal),axis=1)))
                feature = self.extractor.extract_stft(signal)
                # print("mean",np.mean(np.mean((feature),axis=1)))
                # print(np.max(np.amax((feature),axis=1)))  
                norm_feature = self.normalizer.normalize(feature)
                # print(np.max(np.amax((norm_feature),axis=1)))         
                # np_array - source - train/val/test - j,k (track/sgement)
                if source == "mixture":
                    save_file = self.saver.save_feature(norm_feature,source,self.dataset_type,j,k,aug,inference=inference) # only mixture can use inference mode
                    self._store_min_max(save_file,feature.min(),feature.max())       
                if source == "vocals":
                    save_file = self.saver.save_feature(norm_feature,source,self.dataset_type,j,k,aug,)
                    self._store_min_max(save_file,feature.min(),feature.max())       
                if source == "bass":
                    save_file = self.saver.save_feature(norm_feature,source,self.dataset_type,j,k,aug)
                    self._store_min_max(save_file,feature.min(),feature.max())   
                if source == "drums":
                    save_file = self.saver.save_feature(norm_feature,source,self.dataset_type,j,k,aug)
                    self._store_min_max(save_file,feature.min(),feature.max())            
                if source == "other":
                    save_file = self.saver.save_feature(norm_feature,source,self.dataset_type,j,k,aug)
                    self._store_min_max(save_file,feature.min(),feature.max())       
                if source == "accompaniment":
                    save_file = self.saver.save_feature(norm_feature,source,self.dataset_type,j,k,aug)
                    self._store_min_max(save_file,feature.min(),feature.max())       
                if not inference: print(f"processed file {save_file}",end="\r") 
          

    def _is_padding_neccessary(self,signal):
        if len(signal) < self._num_expected_samples: #-len(signal) (is wwrong)
            return True
        return False
    def _apply_padding(self,signal):
        num_missing_samples = self._num_expected_samples - len(signal)        
        padded_signal = self.padder.right_pad(signal,num_missing_samples)
        return padded_signal

    def _store_min_max(self,save_path,min_value,max_value):
        if max_value > self.maximum_val:            
            self.maximum_val = max_value
        if min_value < self.minimum_val:
            self.minimum_val = min_value

        self.min_max_values[save_path] = {
           "min": min_value,
           "max": max_value,
       }
       

def init_Preprocessing():
    loader = Loader(SAMPLE_RATE,MONO)
    padder = Padder("constant")
    log_spectrogram_extractor = LogSpectroGramExtractor(N_FFT)
    min_max_normalizer = MinMaxNormalizer(0,1)
    augmentor = Augmentation(AUGMENT)
    saver = Saver(MIN_MAX_VALUES_SAVE_DIR)

    preprocessing = PreprocessingPipeline(chunk_duration=CHUNK_DURATION)
    preprocessing.loader = loader
    preprocessing.augmentor = augmentor
    preprocessing.padder = padder
    preprocessing.extractor = log_spectrogram_extractor
    preprocessing.normalizer = min_max_normalizer
    preprocessing.saver = saver
    return preprocessing

def main():
    preprocessing = init_Preprocessing()
    # preprocessing.process_database()
    preprocessing.process_input_track()

if __name__=="__main__":
    main()