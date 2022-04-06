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
        self.filelist = glob.glob(os.path.join("F:/Thesis/train/mixture", '*'))
        self.filelist.sort(key=natural_keys)
        self.len_train_data = len(self.filelist)
        self.nr_batches = int(self.len_train_data/self.batch_size) - 50 #-50 is because currently end songs are buged.
        print("total files:\t",self.len_train_data)
        print("batch_size:\t",self.batch_size)
        print("total nr batches:\t",self.nr_batches)

    def load_data(self,batch_nr):
        
        # make sure we shuffle self.filelist!
        # def shuffle():
        
    
        # total_valid 
        # total_test

  

        # print("total files:\t",self.len_train_data)
        # print("batch_size:\t",self.batch_size)
        # print("total nr batches:\t",self.nr_batches)
        # print(5/0)
        global shape
        
            
        x_train = []
        y_train = []
        for i in range(batch_nr,batch_nr+self.batch_size): # all batches we want to take
            
            global counter
            # print(counter)
            file = self.filelist[counter]
            normalized_spectrogram = np.load(file)
            if counter == 0:
               
                shape = list(normalized_spectrogram.shape)
                # print(shape,"shape")
                
            if list(normalized_spectrogram.shape) == shape: # if not, then padding did not work correcty!
                x_train.append(normalized_spectrogram)
                # print(self.filelist[counter])
            counter+=1
        
        x_train = np.array(x_train)   
        # print(__name__)
        # print(x_train.shape,counter)  
        return x_train,x_train

# gen = DataLoader(batch_size=8,num_epoch=10)
# gen.load_data(16)
# for i in gen:
#     print(i.shape)
# next(gen)
# print(gen)
