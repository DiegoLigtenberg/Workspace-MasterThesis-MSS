import numpy as np
from mss.utils.dataloader import natural_keys, atof
from mss.postprocessing.generator_c import *
from mss.preprocessing.preprocesssing_main import *
import glob
import os
import pickle
'''
For each track in track_input

1) Load track
  - load IRMAS DATABASE
  - load MUSDB DATABASE
2) preprocess the track
  - preprocess IRMAS DATABASE
  - (MUSDB already preprocessed)
3) Model the track
  - convert preprocessed data into waveforms (track wise)
  - name the generated tracks according to original track_names
4) Evaluate tracks
'''

TEST_DATA_DIR = "G:/Thesis/test/mixture/"

class EncodedTestData():
    def __init__(self, save=False):
        self.save = save
        self.filelist_X, self.filelist_Y = self.load_musdb()

    def load_musdb(self):
        def save_pickle():
            filelist_X = glob.glob(os.path.join(f"{TEST_DATA_DIR}mixture", '*'))
            filelist_X.sort(key=natural_keys)
            filelist_X = filelist_X[0::]
            filelist_Y = glob.glob(os.path.join(f"{TEST_DATA_DIR}other", '*'))
            filelist_Y.sort(key=natural_keys)
            filelist_Y = filelist_Y[0::]

            filelist_X_new = []
            file_list_Y_new = []

            current_song = -1
            for i, song in enumerate(filelist_X):
                chunks_X = []
                chunks_Y = []
                if current_song != int(filelist_X[i].split("\\")[1].split("-")[0]):
                    current_song = int(
                        filelist_X[i].split("\\")[1].split("-")[0])
                    for chunk in filelist_X:
                        if int(chunk.split("\\")[1].split("-")[0]) == current_song:
                            chunks_X.append(chunk)
                    for chunk in filelist_Y:
                        if int(chunk.split("\\")[1].split("-")[0]) == current_song:
                            chunks_Y.append(chunk)
                    filelist_X_new.append(chunks_X)
                    file_list_Y_new.append(chunks_Y)
            with open('mss_evaluate_data/test_X.pkl', 'wb') as f:
                print("files saved to:\t", "mss_evaluate_data")
                pickle.dump(filelist_X_new, f)
            with open('mss_evaluate_data/test_Y.pkl', 'wb') as f:
                pickle.dump(file_list_Y_new, f)
        if self.save:
            save_pickle()
        filelist_X = pickle.load(open("mss_evaluate_data/test_X.pkl", "rb"))
        filelist_Y = pickle.load(open("mss_evaluate_data/test_Y.pkl", "rb"))
        return filelist_X, filelist_Y

class Separator():
    '''
    separator is able to convert input songs to predicted waveforms 
    takes as input x_mixture , y_other'''
    
    def __init__(self) -> None:
        pass

    def gen_eval(self): 
        while self.song_iterator < self.file_length:
            self.song_iterator+=1
            yield self.filelist_X[self.song_iterator], self.filelist_Y[self.song_iterator] 

    def evaluate_musdb_test(self):
        '''
        inp: converts npy spectrograms (encoded test data) to evaluation metrics
        '''
        self.filelist_X,self.filelist_Y = EncodedTestData(save=False).load_musdb()
        self.file_length = len(self.filelist_X)
        self.song_iterator = -1 # song is 1 lower than when you start songs counting from 1 !   
        gen = Generator()
        for i in range(self.file_length):    
            x_mixture_file_chunks,y_target_file_chunks = next(self.gen_eval())                
            gen.generate_waveform(x_mixture_file_chunks,y_target_file_chunks,self.song_iterator,inference=False,save_mixture=False)
        print("done")

    def input_to_waveform(self):
        self.preprocessor = init_Preprocessing()
        for i in range(3):
            try:
                next(self.preprocessor.proces_input_track_generator())
                print("worked")
            except Exception as e:
                print(type(e))
        # load input track
        # convert to chunks
        # save as npy spectrogram
        # separate
        # save waveform

        pass



separator = Separator()
separator.input_to_waveform()
# separator.evaluate_musdb_test()
