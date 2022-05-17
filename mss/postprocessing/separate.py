from mss.utils.dataloader import natural_keys, atof
from mss.postprocessing.generator_c import *
from mss.preprocessing.preprocesssing_main import *
from tqdm import tqdm
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

class EncodedSpectrograms():
    '''returns directory of encoded spectrograms in a list'''
    def __init__(self):
        self.filelist_X, self.filelist_Y = self.load_musdb()

    def load_musdb(self,save=False):
        def save_pickle():
            filelist_X = glob.glob(os.path.join(f"{TEST_DATA_DIR}mixture", '*'))
            filelist_X.sort(key=natural_keys)
            filelist_Y = glob.glob(os.path.join(f"{TEST_DATA_DIR}other", '*'))
            filelist_Y.sort(key=natural_keys)

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
        if save:
            save_pickle()
        filelist_X = pickle.load(open("mss_evaluate_data/test_X.pkl", "rb"))
        filelist_Y = pickle.load(open("mss_evaluate_data/test_Y.pkl", "rb"))
        return filelist_X, filelist_Y
    
    def load_inference(self):
        filelist_X = glob.glob(os.path.join(f"{TEMP_INFERENCE_SAVE_DIR}", '*'))
        filelist_X.sort(key=natural_keys)        
        return filelist_X
        
class Separator():
    '''
    separator is able to convert input songs to predicted waveforms 
    takes as input x_mixture , y_other'''
    
    def __init__(self,post_processing=True) -> None:
        self.post_processing = post_processing
        self.encoded_spectrograms = EncodedSpectrograms()        
        self.gen = Generator(post_processing=post_processing)

    def _gen_eval(self): 
        while self.song_iterator < self.file_length:
            self.song_iterator+=1
            yield self.filelist_X[self.song_iterator], self.filelist_Y[self.song_iterator] 

    def evaluate_musdb_test(self):
        '''
        inp: converts npy spectrograms (encoded test data) to evaluation metrics
        '''
        self.filelist_X,self.filelist_Y = self.encoded_spectrograms.load_musdb(save=False)
        self.file_length = len(self.filelist_X)
        self.song_iterator = -1 # song is 1 lower than when you start songs counting from 1 !   
        
        for i in tqdm(range(self.file_length)):    
            x_mixture_file_chunks,y_target_file_chunks = next(self._gen_eval())                
            self.gen.generate_waveform(x_mixture_file_chunks,y_target_file_chunks,self.song_iterator,inference=False,save_mixture=False)
        print("done")

    def input_to_waveform(self):
        self.preprocessor = init_Preprocessing()
        
        # delete temp folder if it exists
        try: self._delete_temp()
        except FileNotFoundError as e: print(e)

        # for all songs in input directory: use mss loop
        for i in (range(self.preprocessor.loader.input_track_len)):
            # encode the song as spectrogram
            file_name = ""
            try: file_name = next(self.preprocessor.proces_input_track_generator()) 
            except StopIteration as e: break # stops running all code beneath

            # use generator to generate and save waveform
            x_mixture = self.encoded_spectrograms.load_inference()
            self.gen.generate_waveform(x_mixture_file_chunks=x_mixture,inference=True,save_mixture=False,file_name=file_name)            
            
            # delete save dir spectrogram
            self._delete_temp()

    def _delete_temp(self):
        for f in os.listdir(TEMP_INFERENCE_SAVE_DIR):
            assert f[-4:] == ".npy", f"illegal file extension found: {f[-4:]} file.\t can only handle .npy files."
            os.remove(f"{TEMP_INFERENCE_SAVE_DIR}/{f}")
        assert TEMP_INFERENCE_SAVE_DIR == "temp_inference", f"INFERENCE SAVE DIR MAY NOT BE CHANGED"
        os.rmdir(TEMP_INFERENCE_SAVE_DIR)

def main():
    separator = Separator()
    separator.input_to_waveform()
    # separator.evaluate_musdb_test()

if __name__=="__main__":
    main()
