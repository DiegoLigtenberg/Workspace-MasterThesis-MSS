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
shape = None   
class DataLoader():
    def __init__(self,batch_size,num_epoch) -> None:
        self.batch_size = batch_size
        self.num_epoch = num_epoch

        extensions = ["mixture","vocals","bass","drums","other","accompaniment"]
        self.filelist_X = glob.glob(os.path.join("G:/Thesis/train/mixture", '*'))
        self.filelist_X.sort(key=natural_keys)
        self.filelist_X = self.filelist_X #[0:10]
        
     
        self.filelist_Y = glob.glob(os.path.join("G:/Thesis/train/vocals", '*'))
        self.filelist_Y.sort(key=natural_keys)        
        self.filelist_Y = self.filelist_Y #[0:10]

        self.shuffle_data()
        # print(self.filelist_X[0:2],self.filelist_Y[0:2])
        # merge = zip(self.filelist_X,self.filelist_Y)
        # self.filelist_X, self.filelist_Y = zip(*merge)

        # random.shuffle(self.filelist_X)
        # random.shuffle(self.filelist_X)

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

    def shuffle_data(self):
        merge = list(zip(self.filelist_X,self.filelist_Y))
        random.shuffle(merge)
        self.filelist_X, self.filelist_Y = zip(*merge)
        # print(self.filelist_X[0:2],self.filelist_Y[0:2])

    def reset_counter(self):
        global counter
        counter = 0 
gen = DataLoader(batch_size=1,num_epoch=10)
gen.load_data(1)
# for i in gen:
#     print(i.shape)
# next(gen)
# print(gen)
