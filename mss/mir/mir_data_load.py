import glob
import os
import pandas as pd
import numpy as np
import keras
from collections import OrderedDict
from mss.utils.dataloader import natural_keys
from mss.mir.mir_settings import SPECTROGRAM_PATHS

class MIR_DataLoader():
    '''
        this class is responsible for loading the 
        - input:    dictionary of path directories of cached spectrograms (.npy): 
                    dictionary of labels path directory corresponding to cached spectrograms
        - output:   list of spectrogram path file names
                    labels corresponding to spectrogram path file names as .npy 
        labels of spectrogram path files from a directory and outputs their file names
        - labels corresponding to the cached spectrograms
    '''

    def __init__(self,verbose=True):
        self.verbose = verbose
        self._class_weights = None

    def load_data(self,dataset,model):
        '''
        this class is responsible for loading the 
        - input:    dictionary of path directories of cached spectrograms (.npy): 
                    dictionary of labels path directory corresponding to cached spectrograms
        - output:   list of spectrogram path file names
                    labels corresponding to spectrogram path file names as .npy 
        labels of spectrogram path files from a directory and outputs their file names
        - labels corresponding to the cached spectrograms
        '''
        x_train,y_train = self._create_data(dataset,model)
        return x_train,y_train
    
    @classmethod
    def get_train_class_weights(cls):
        '''calculates class weights for (imbalanced) train dataset according to: https://www.analyticsvidhya.com/blog/2020/10/improve-class-imbalance-class-weights'''
        train_label_path = SPECTROGRAM_PATHS["train"]["labels"]
        class_samples = pd.read_csv(train_label_path).sum()
        amnt_classes = len(class_samples)
        total_samples = sum(class_samples)        
        train_class_weights = enumerate([total_samples/(amnt_classes* x) for x in class_samples])
        train_class_weights_d = dict((k,v) for k,v in train_class_weights)
        return train_class_weights_d

    def _create_data(self,dataset,model):
        # load the path of spectrograms from specified data split (train/val/test) and model (base/no_post/with_post)
        assert dataset in SPECTROGRAM_PATHS.keys(), f"specify dataset with: 'train' - 'val' - 'test'"
        assert model in SPECTROGRAM_PATHS[list(SPECTROGRAM_PATHS.keys())[0]].keys(), f"specify model: 'base' - 'no_postprocessing' - 'with_postprocessing'"        
        x_path = SPECTROGRAM_PATHS[dataset][model]
        y_path = SPECTROGRAM_PATHS[dataset]["labels"]
        assert y_path != None, f"check if name of label directory is not changed"
        
        if self.verbose: print(x_path)

        data = glob.glob(os.path.join(x_path, '*'),recursive=True)
        data.sort(key=natural_keys)

        x_train = []  
        for i,file in enumerate(data):
            x_train.append(file)

        y_train = []
        y_train = pd.read_csv(y_path).to_numpy()

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        if self.verbose:print(f"loaded dataset: {dataset} - model: {model} ",x_train.shape,y_train.shape)
        return x_train,y_train


class My_Custom_Generator(keras.utils.all_utils.Sequence):
    '''
    input:  (X_file names, Y labels)
    output: (X_file.npy, Y_labels)
    '''
    def __init__(self,image_file_names,labels,batch_size):
        self.image_file_names = image_file_names
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_file_names) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self,idx):
        batch_x = self.image_file_names[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx*self.batch_size : (idx+1) * self.batch_size]
        return np.array([
            (np.load(str(file_name)))
               for file_name in batch_x]), np.array(batch_y)

if __name__== "__main__":   
    loader = MIR_DataLoader()
    loader.load_data(dataset="test",model="with_postprocessing")
    print(MIR_DataLoader.get_train_class_weights())