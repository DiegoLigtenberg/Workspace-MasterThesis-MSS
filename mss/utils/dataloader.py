import glob
import re 
import numpy as np
import os
import random



def atof(text):
    try:
        retval = float(text)
    except ValueError:
        retval = text
    return retval

def natural_keys(text):
    '''
    alist.sort(key=natural_keys) sorts in human order
    http://nedbatchelder.com/blog/200712/human_sorting.html
    (See Toothy's implementation in the comments)
    float regex comes from https://stackoverflow.com/a/12643073/190597
    '''
    return [ atof(c) for c in re.split(r'[+-]?([0-9]+(?:[.][0-9]*)?|[.][0-9]+)', text) ]

     
counter = 0 
counter_val = 0
shape = None   
class DataLoader():
    def __init__(self,batch_size,num_epoch) -> None:
        self.batch_size = batch_size
        self.num_epoch = num_epoch

        extensions = ["mixture","vocals","bass","drums","other","accompaniment"]
        self.filelist_X = glob.glob(os.path.join("G:/Thesis/train/mixture", '*'))
        self.filelist_X.sort(key=natural_keys)
        self.filelist_X = self.filelist_X[0::]
     
        self.filelist_Y = glob.glob(os.path.join("G:/Thesis/train/other", '*'))
        self.filelist_Y.sort(key=natural_keys)        
        self.filelist_Y = self.filelist_Y[0::]

        self.filelist_X_V = glob.glob(os.path.join("G:/Thesis/valid/mixture", '*'))
        self.filelist_X_V.sort(key=natural_keys)
        self.filelist_X_V = self.filelist_X_V[0::]
     
        self.filelist_Y_V = glob.glob(os.path.join("G:/Thesis/valid/other", '*'))
        self.filelist_Y_V.sort(key=natural_keys)        
        self.filelist_Y_V = self.filelist_Y_V[0::]
        
        self.outliers_train = []
        self.outliers_val = []

        self.shuffle_data()

        self.len_train_data = len(self.filelist_X)
        self.nr_batches = int(self.len_train_data/self.batch_size)  #-50 is because currently end songs are buged.
        print("total files:\t",self.len_train_data)
        print("batch_size:\t",self.batch_size)
        print("total nr batches:\t",self.nr_batches)

    def load_data(self,batch_nr):
        global shape
        global counter
        # if batch_nr >= self.len_train_data-16:
        #     print(counter)
        #     counter = 0
        x_train = []
        y_train = []

        '''SHOULD REMOVE -8 this  is weird???'''
        for i in range(batch_nr,batch_nr+self.batch_size): # all batches we want to take
            # print(counter)
            # print(batch_nr)
            file_X = self.filelist_X[counter]
            file_Y = self.filelist_Y[counter]
            normalized_spectrogram_X = np.load(file_X)
            normalized_spectrogram_Y = np.load(file_Y)
            if counter == 0:               
                shape = list(normalized_spectrogram_X.shape)
                
            if list(normalized_spectrogram_X.shape) == shape: # if not, then padding did not work correcty!
                x_train.append(normalized_spectrogram_X)
                y_train.append(normalized_spectrogram_Y)
            # if counter <4:
            counter+=1

        
        x_train = np.array(x_train)   
        y_train = np.array(y_train)
        return x_train,y_train

    def load_val(self,batch_nr):
        global counter_val
        # if batch_nr >= self.len_train_data-16:
        #     print(counter)
        #     counter = 0
        x_val = []
        y_val = []

        '''SHOULD REMOVE -8 this  is weird???'''
        for i in range(batch_nr,batch_nr+self.batch_size): # all batches we want to take
            file_X = self.filelist_X_V[counter_val]
            file_Y = self.filelist_Y_V[counter_val]
            normalized_spectrogram_X = np.load(file_X)
            normalized_spectrogram_Y = np.load(file_Y)
                
            if list(normalized_spectrogram_X.shape) == shape: # if not, then padding did not work correcty!
                x_val.append(normalized_spectrogram_X)
                y_val.append(normalized_spectrogram_Y)
            # if counter <4:
            counter_val+=1

        
        x_val = np.array(x_val)   
        y_val = np.array(y_val)
        return x_val,y_val

    def showcase_outlier_train(self):        
        global counter
        self.outliers_train.append(counter)
        print(f"train outlier:\t {self.filelist_X[counter]}, counterval:\t{counter}")
        print(f"train outlier_Y:\t {self.filelist_Y[counter]}, counterval:\t{counter}")
        print("")
        print(self.outliers_train)


    def showcase_outlier_val(self):    
        global counter_val    
        self.outliers_val.append(counter_val)
        print(f"validation outlier:\t {self.filelist_X_V[counter_val]}, counterval:\t{counter_val}")
        print(f"validation outlier_Y:\t {self.filelist_Y_V[counter_val]}, counterval:\t{counter_val}")
        print("")
        print(self.outliers_val)


    def shuffle_data(self):
        # shuffle train data
        merge = list(zip(self.filelist_X,self.filelist_Y))
        random.shuffle(merge)
        self.filelist_X, self.filelist_Y = zip(*merge)

        # shuffle val data
        merge = list(zip(self.filelist_X_V,self.filelist_Y_V))
        random.shuffle(merge)
        self.filelist_X_V, self.filelist_Y_V = zip(*merge)

        # print(self.filelist_X[0:2],self.filelist_Y[0:2])

    def reset_counter(self):
        global counter
        global counter_val
        counter = 0 
        counter_val = 0
# gen = DataLoader(batch_size=1,num_epoch=10)
# gen.load_data(1)
# for i in gen:
#     print(i.shape)
# next(gen)
# print(gen)
