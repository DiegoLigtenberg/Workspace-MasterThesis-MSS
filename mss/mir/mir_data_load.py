import glob
import os
import pandas as pd
import numpy as np
import keras
from mss.utils.dataloader import natural_keys

class MIR_DataLoader():

    def __init__(self,verbose=True):
        self.train = None
        self.model = None
        self.paths = None
        self.verbose = verbose

    def load_data(self,train,model):
        '''
        inp train:      - train OR test
        inp model:      - base, no_postprocessing, with_postprocessing
        '''
        self.train = train
        self.model = model
        if train == "train":
            if model == "base":                     self.paths = 0
            if model == "no_postprocessing":        self.paths = 1
            if model == "with_postprocessing":      self.paths = 2
        if train == "test":
            if model == "base":                     self.paths = 3
            if model == "no_postprocessing":        self.paths = 4
            if model == "with_postprocessing":      self.paths = 5
        assert self.paths != None, "specify train or test with base, no_postprocessing, with_postprocessing"
        
        x_train,y_train = self._create_data()
        return x_train,y_train


    def _create_data(self):
        spectrogram_paths = [
            "G:/Thesis/MIR_datasets/train_dataset/spectrogram_base", #MIR_datasets/train_dataset/spectrogram_base",
            "G:/Thesis/MIR_datasets/train_dataset/spectrogram_no_post",#"MIR_datasets/train_dataset/spectrogram_no_post",
            "G:/Thesis/MIR_datasets/train_dataset/spectrogram_with_post", #"MIR_datasets/train_dataset/spectrogram_with_post",

            "G:/Thesis/MIR_datasets/test_dataset/spectrogram_base", #"MIR_datasets/test_dataset/spectrogram_base",
            "G:/Thesis/MIR_datasets/test_dataset/spectrogram_no_post",#"MIR_datasets/test_dataset/spectrogram_no_post",
            "G:/Thesis/MIR_datasets/test_dataset/spectrogram_with_post"#"MIR_datasets/test_dataset/spectrogram_with_post",
        ]
        paths = spectrogram_paths[self.paths]
        if self.verbose: print(paths)

        data = glob.glob(os.path.join(paths, '*'),recursive=True)
        data.sort(key=natural_keys)

        x_train = []  
        for i,file in enumerate(data):
            x_train.append(file)

        y_train = []
        if self.train == "train":
            y_train = pd.read_csv("MIR_datasets/MIR_train_labels_merged.csv").to_numpy()
        if self.train == "test":
            y_train = pd.read_csv("MIR_datasets/MIR_test_labels.csv").to_numpy()

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        if self.verbose:print(f"Loaded data: {self.train} - {self.model} ",x_train.shape,y_train.shape)
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
    loader.load_data(train="test",model="base")
