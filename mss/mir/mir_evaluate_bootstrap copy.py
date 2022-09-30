
import numpy as np
import pandas as pd

# datasize is extracted from real source with similar sounding excerpts that is 4 times as large
segment_size = 3 
data_size =  (5+20)/2 #=6
data_size = 6 #data_size/segment_size 

columns=["cel","cla","flu","gac","gel","org","pia","sax","tru","vio","voi"]
def calculate_CI_AUC(cur_auc,n1,n2):
    auc = cur_auc
    N1 = n1*data_size #N1 = 527
    N2 = (n2*data_size)-N1 #N2 = 279                            #totalinstruments - segments containing this instrument
    q0 = auc*(1-auc)
    q1 = auc / (2-auc) - auc**2
    q2 =  2*auc**2/(1+auc) - auc **2
    se =   np.sqrt( (q0 + (N1-1)*q1 + (N2-1)*q2 )     / (N1*N2))
    z = 1.95
    upper = round(auc + z*se,3) 
    lower = round(auc - z*se,3)
    print("auc ",round(auc,3),"\tupper ",upper,"\tlower ",lower,"\terror ",np.round(se*z,3))
    return se*z
    

def myfunc(model="base"):
    print(model)
    df = pd.read_csv("MIR_datasets/MIR_test_labels_combined.csv") #test dataset


    if model == "base": model = r"E:/Documenten E/University/Jaar 5/Project MSS/Workspace-MasterThesis-MSS/visualisation/MIR/base/ROC_Curves/_instrument_aucs.txt"
    if model == "raw":  model = r"E:/Documenten E/University/Jaar 5/Project MSS/Workspace-MasterThesis-MSS/visualisation/MIR/no_postprocessing_regular/ROC_Curves/_instrument_aucs.txt"
    if model == "postprocessing":  model = r"E:/Documenten E/University/Jaar 5/Project MSS/Workspace-MasterThesis-MSS/visualisation/MIR/with_postprocessing/ROC_Curves/_instrument_aucs.txt"

    with open (model) as f:
        lines = f.readlines()        
        auc_list = []
        se = []
        for i,line in enumerate(lines):
            n1  = list(df.sum())[i]
            n2 =  sum(df.sum()) 
            split_row = line.split('.')
            if len(split_row) > 1:
                len_line =  len(split_row[1])-1
                auc = float(split_row[1])/(10**len_line) # divided by amount of float values after .
                print(columns[i],"",end="")
                se.append(calculate_CI_AUC(auc,n1,n2))
                auc_list.append(auc)
        print(np.mean(auc_list),"\n")
        f.close()
    return auc_list,se


print(pd.read_csv("MIR_datasets/MIR_test_labels_combined.csv").sum()) #test dataset
base_aucs, base_se = myfunc("base")
raw_aucs, raw_se = myfunc("raw")
postprocess_aucs, postprocess_se = myfunc("postprocessing")
    

a = base_aucs 
a = [sum(x) for x in zip(base_aucs, raw_se,base_se)]
b = raw_aucs
raw = [ round(b_i - a_i,3) for a_i, b_i in zip(a, b)]


a = base_aucs 
a = [sum(x) for x in zip(base_aucs, postprocess_se,base_se)]
b = postprocess_aucs
post = [ round(b_i - a_i,3) for a_i, b_i in zip(a, b)]

merge = []
for i,val in enumerate(post):
    if val == False and raw[i] ==False: merge.append(True)
    else:merge.append(False)
print("raw")
print(raw)
print("postprocess")
print(post)
print("merge")
print(merge)


# print(b,"\n",a)
# b = raw_aucs+raw_se+base_se 
# print(len(b))
# print(len(a))


# print(np.subtract(raw_aucs,base_aucs+raw_se+base_se  ))

def calculate_CI_AUC_mean(cur_auc,n1,n2):
    auc = cur_auc
    N1 = n1*data_size #N1 = 527
    N2 = n2*data_size #N2 = 279                            #totalinstruments - segments containing this instrument
    q0 = auc*(1-auc)
    q1 = auc / (2-auc) - auc**2
    q2 =  2*auc**2/(1+auc) - auc **2
    se =   np.sqrt( (q0 + (N1-1)*q1 + (N2-1)*q2 )     / (N1*N2))
    z = 1.95
    upper = round(auc + z*se,3) 
    lower = round(auc - z*se,3)
    print("auc ",round(auc,3),"\tupper ",upper,"\tlower ",lower,"\terror ",np.round(se*z,3))

calculate_CI_AUC_mean(0.786,1279,1279)

# [False, True, True, False, True, True, False, True, True, False, False]