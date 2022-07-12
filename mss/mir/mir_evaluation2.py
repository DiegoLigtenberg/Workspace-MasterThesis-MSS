
from mss.mir.mir_conv_net import ConvNet
from mss.mir.mir_data_load import MIR_DataLoader, My_Custom_Generator
from mss.mir.mir_train import existing_model

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics  import f1_score
from sklearn.metrics import multilabel_confusion_matrix

import numpy as np

class Evaluate():
    pass


if __name__ == "__main__":
    model_name = "no_post_last_ auc 0._1_1_1_33 auc 0.7462" # 100 epoch
    # model_name = "no_post_less_regular_noWI_73 auc 0.7644" # 100 epoch
    
    conv_net = existing_model(model_name)
    dataloader = MIR_DataLoader()

    x_train = np.load("G:\Thesis\inference\mixture\inference-0-0-0.npy")
    x_train = x_train[np.newaxis,:,:]
    # print("poop",x_train.shape)
    # asd
    optimal_tresholds = [0.0427913 ,0.0017562, 0.0007017 ,0.0162022, 0.126483,  0.0060873, 0.0640982 ,0.0173643, 0.0245965 ,0.0235464, 0.332417 ]
    preds = conv_net.model.predict(x_train)
    [preds] = preds
    print(np.round(preds,2))
    # asd
    preds = np.array(preds)
    # print(preds)
    # print(preds[3])
    # asd
    # optimal_tresholds = [0.2] *11
    for j in range(11):
        preds[j] = (preds[j]> optimal_tresholds[j]) #.astype(float)
    # preds = preds[0]
    # print(preds)
    # [preds] = preds # only take first element in batch
    # print(preds)
    # from scipy.special import expit
    # print(preds)
    # preds = expit(preds)
    # preds = [round(num,2) for num in preds]
    print(preds,"\n")
    print(preds.shape)
    asd