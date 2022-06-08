import glob
import os
import pandas as pd
import numpy as np
from sqlalchemy import null
from mss.utils.dataloader import natural_keys

class MIR_DataLoader():

    def __init__(self):
        self.train = None
        self.model = None
        self.paths = None

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
            "MIR_datasets/train_dataset/spectrogram_base",
            "MIR_datasets/train_dataset/spectrogram_no_post",
            "MIR_datasets/train_dataset/spectrogram_with_post",

            "MIR_datasets/test_dataset/spectrogram_base",
            "MIR_datasets/test_dataset/spectrogram_no_post",
            "MIR_datasets/test_dataset/spectrogram_with_post",
        ]
        paths = spectrogram_paths[self.paths]
        print(paths)

        data = glob.glob(os.path.join(paths, '*'),recursive=True)
        data.sort(key=natural_keys)

        x_train = []  
        for i,file in enumerate(data):
            x_train.append(file)

        y_train = []
        if self.train == "train":
            y_train = pd.read_csv("MIR_datasets/MIR_train_labels.csv").to_numpy()
        if self.train == "test":
            y_train = pd.read_csv("MIR_datasets/MIR_test_labels.csv").to_numpy()

        x_train = np.array(x_train)
        y_train = np.array(y_train)
        print(f"Loaded data: {self.train} - {self.model} ",x_train.shape,y_train.shape)
        return x_train,y_train

if __name__== "__main__":
    loader = MIR_DataLoader()
    loader.load_data(train="train",model="with_postprocessing")
