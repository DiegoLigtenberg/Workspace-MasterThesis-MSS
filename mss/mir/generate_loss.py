from email.mime import base
import numpy as np
import os
from mss.utils.dataloader import natural_keys
from mss.utils.visualisation import visualize_loss

# generate loss for no_posprocessing
def no_postprocessing_auc():
    with open ("MIR_saved_models/mir_model_no_postprocessing_76 auc 0.7333/training logs.txt","r") as f:
        lines = f.read().splitlines()

    with open ("MIR_trained_models\mir_model_no_postprocessing_ auc 0._1_1_1_19 auc 0.7405/model logs epoch 100-300.txt","r") as f2:
        lines.extend(f2.read().splitlines())

    my_loss = []
    for line in lines:
        if "m_auc" in line:
            
            my_loss.append(float(line.split("m_auc:")[1] ))

    my_loss = np.array(my_loss)
    my_loss = my_loss[:300]
    print(my_loss.shape)

    visualize_loss(total_train_loss=my_loss,total_val_loss=my_loss,smoothing=10,model_name="mir_no_postprocessing_model",save=True,)

def with_postprocessing_auc():
    with open ("MIR_saved_models/mir_model_with_postprocessing_98 auc 0.7019/model loss logs.txt","r") as f:
        lines = f.read().splitlines()

    with open ("MIR_saved_models/mir_model_with_postprocessing_ auc 0.0_1_1_100 auc 0.74/model loss logs epoch 98-198.txt","r") as f2:
        lines.extend(f2.read().splitlines())
        
    with open ("MIR_saved_models/mir_model_with_postprocessing_ auc 0.0_00 auc 0._1_1_1_59 auc 0.745/model logs.txt","r") as f2:
        lines.extend(f2.read().splitlines())

    my_loss = []
    for line in lines:
        if "m_auc" in line:
            
            my_loss.append(float(line.split("m_auc:")[1] ))

    my_loss = np.array(my_loss)
    my_loss = my_loss[::]
    print(my_loss.shape)
    # asd

    visualize_loss(total_train_loss=my_loss,total_val_loss=my_loss,smoothing=10,model_name="mir_with_postprocessing_model",save=True,)

def base_auc():
    files = os.listdir('MIR_trained_models')
    files.sort(key=natural_keys)
    my_loss = []
    for file in files:
        if "mir_model_base" in file:
            my_loss.append(float(file.split("auc")[1]))

    my_loss = np.array(my_loss)

    print(my_loss)
    visualize_loss(total_train_loss=my_loss,total_val_loss=my_loss,smoothing=5,model_name="mir_base_model",save=True,)

if __name__=="__main__":
    # base_auc()

    # no_postprocessing_auc()
    with_postprocessing_auc()