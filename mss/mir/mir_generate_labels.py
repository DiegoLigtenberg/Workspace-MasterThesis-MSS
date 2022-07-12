
import glob
import os
import pandas as pd
import numpy as np
from mss.utils.dataloader import natural_keys

PATHS = ["MIR_datasets/train_dataset/track_output_base","F:\Thesis\instr classification dataset\IRMAS-TestingData-Part3"] # paths to og files of irmas dataset

# calcualte class weights
df = pd.read_csv("MIR_datasets/MIR_test_labels_combined.csv")

b = df.sum()

n_samples = sum(b)

weights = [n_samples/(11* x) for x in b]

print(b)
print(weights)

asd

class LabelGenerator():
    '''
    class looks at the original train data file, and the original test data file for irmas dataset and generates labels in pandas dataframe format.
    '''
    def __init__(self,paths,train,save=False):
        path = None
        data = None
        if train: path = paths[0];data = glob.glob(os.path.join(path, '*/*.wav'),recursive=True)
        else: path = paths[1];data = glob.glob(os.path.join(path, '*/*'),recursive=True)            
        data.sort(key=natural_keys)    
        instrument_to_val = {   "\\cel\\": 0, 
                                "\\cla\\": 1,
                                "\\flu\\": 2,
                                "\\gac\\": 3,
                                "\\gel\\": 4,
                                "\\org\\": 5,
                                "\\pia\\": 6,
                                "\\sax\\": 7,
                                "\\tru\\": 8,
                                "\\vio\\": 9,
                                "\\voi\\": 10,
        }
        instrument_to_val_test = {  "cel":0, 
                                    "cla":1,
                                    "flu":2,
                                    "gac":3,
                                    "gel":4,
                                    "org":5,
                                    "pia":6,
                                    "sax":7,
                                    "tru":8,
                                    "vio":9,
                                    "voi":10,
        }
        instrument_count_matrix = np.zeros(11)
        columns=["cel","cla","flu","gac","gel","org","pia","sax","tru","vio","voi"]

        # generate train data labels
        if train:
            for file in data:        
                for instr_str in instrument_to_val.keys():
                    if instr_str in file:
                        row = np.zeros(11)
                        row[instrument_to_val[instr_str]] +=1
                        instrument_count_matrix = np.vstack([instrument_count_matrix,row])
            instrument_count_matrix = np.delete(instrument_count_matrix,(0),axis=0) # delete initialisation row
            df = pd.DataFrame(instrument_count_matrix,columns=columns)

        # generate test data labels
        else:
            for i,file in enumerate(data):
                if file[-4:] == ".txt":
                    test_instr = []
                    with open (f"{file}","r") as myfile:
                        test_instr = myfile.readlines()
                        test_instr = [w.strip().replace("\t","") for w in test_instr]
                        if "cel" in test_instr:
                            print(i,file,test_instr)
                    row = np.zeros(11)        
                    for word in test_instr:
                        if word in columns:
                            row[instrument_to_val_test[word]] +=1
                    instrument_count_matrix = np.vstack([instrument_count_matrix,row])
            
            instrument_count_matrix = np.delete(instrument_count_matrix,(0),axis=0) # delete initialisation row
            df = pd.DataFrame(instrument_count_matrix,columns=columns)
        
        print(df.sum())
        if save:
            print("saved labels to MIR_datasets")
            if train: df.to_csv("MIR_datasets/MIR_train_labels.csv",index=False)
            else:  df.to_csv("MIR_datasets/MIR_test_labels_combined.csv",index=False)
        

if __name__=="__main__":
    # label_generator_train = LabelGenerator(PATHS,train=True,save=True)
    label_generator_test = LabelGenerator(PATHS,train=False,save=True)


