
import glob
import os
import pandas as pd
import numpy as np
from mss.utils.dataloader import natural_keys, atof
'''
Instrument to Y_target for IRMAS Train 
sigmoid + binary cross entropy
'''


print(396900//44100)

asd
data = glob.glob(os.path.join("F:\Thesis\instr classification dataset\IRMAS-TestingData-Part1", '*/*'),recursive=True)
# data = sorted(data,key=os.path.getmtime)
# data.sort(key=natural_keys)
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

instruments = None
instrument_count_matrix = np.zeros(11)
columns=["cel","cla","flu","gac","gel","org","pia","sax","tru","vio","voi"]

for i,file in enumerate(data):
    
    # txt
    if file[-4:] == ".txt":
        instruments = []
        test_instr = []

        with open (f"{file}","r") as myfile:
            test_instr = myfile.readlines()
            test_instr = [w.strip().replace("\t","") for w in test_instr]
            # print(file,test_instr)
            
        
        row = np.zeros(11)
        # for instr_str in instrument_to_val_test.keys():            
        for word in test_instr:
            if word in columns:
                row[instrument_to_val_test[word]] +=1
        instrument_count_matrix = np.vstack([instrument_count_matrix,row])
        pass
    if file[-4:] == ".wav":
        pass
    # if i>10:
    #     break
instrument_count_matrix = np.delete(instrument_count_matrix,(0),axis=0) # delete initialisation row
# print(instrument_count_matrix)
    # break

df = pd.DataFrame(instrument_count_matrix,columns=columns)

df.to_csv("MIR_test.csv")
# df["sum"] = df.sum(axis=1)

# print(len(df[df["sum"]==4]))
# df

asd
data = glob.glob(os.path.join("track_output_with_post", '*/*.wav'),recursive=True)

# if "\cel\" is in the file dir name -> it means the folder is cello
instrument_to_val = {   "\\cel\\":0, 
                        "\\cla\\":1,
                        "\\flu\\":2,
                        "\\gac\\":3,
                        "\\gel\\":4,
                        "\\org\\":5,
                        "\\pia\\":6,
                        "\\sax\\":7,
                        "\\tru\\":8,
                        "\\vio\\":9,
                        "\\voi\\":10,
}

instrument_count_matrix = np.zeros(11)
for file in data:
    for instr_str in instrument_to_val.keys():
        if instr_str in file:
            row = np.zeros(11)
            row[instrument_to_val[instr_str]] +=1
            instrument_count_matrix = np.vstack([instrument_count_matrix,row])
instrument_count_matrix = np.delete(instrument_count_matrix,(0),axis=0) # delete initialisation row


columns=["cel","cla","flu","gac","gel","org","pia","sax","tru","vio","voi"]
df = pd.DataFrame(instrument_count_matrix,columns=columns)

print(df.iloc)

X = data
Y = df





# for file in data:
#     print(file)
#     break