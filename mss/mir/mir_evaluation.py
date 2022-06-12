
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
    model_name = "mir_model_with_postprocessing_ auc 0.0_1_1_100 auc 0.74" # 100 epoch
    
    conv_net = existing_model(model_name)
    dataloader = MIR_DataLoader()
    x_train, y_train = dataloader.load_data(train="train", model="with_postprocessing")
    x_test, y_test = dataloader.load_data(train="test", model="with_postprocessing")
    # x_train = x_train[::5]
    # y_train = y_train[::5]

    my_train_batch_generator = My_Custom_Generator(x_train, y_train, 1)
    my_test_batch_generator = My_Custom_Generator(x_test, y_test, 1)

    # '''
    preds = conv_net.model.predict(my_test_batch_generator)
   

    
    # '''

    

    # lrap = label_ranking_average_precision_score(y_test,preds)
    # print(lrap)
    # asd
   
    # y_pred = (preds > 0.5)
    # print(y_pred)
    # print(y_pred.shape)
    # f1 = f1_score((y_test>0.02),y_pred,average="macro")
    # print(f1)
    # f1 = f1_score((y_test>0.02),y_pred,average="micro")
    # print(f1)
    # asd

    # f1 = f1_score(y_test,preds)
    # print
    # print(f1)
    # rl = label_ranking_loss(y_test,preds)
    # print(rl)

    # asd
    # print(y_test.shape)
    # print(y_test[:,0].shape)
    # asd
    from sklearn import metrics
    import pandas as pd
    aucs = []
    optimal_tresholds = []
    # fuind optimal cutoffs for test set! such that auc is optimized
    def Find_Optimal_Cutoff(target, predicted):
        fpr, tpr, threshold = roc_curve(target, predicted)
        i = np.arange(len(tpr)) 
        roc = pd.DataFrame({'tf' : pd.Series(tpr-(1-fpr), index=i), 'threshold' : pd.Series(threshold, index=i)})
        roc_t = roc.iloc[(roc.tf-0).abs().argsort()[:1]]    
        return list(roc_t['threshold'])[0] 

    for i in range(11):
        # print(preds[:,0])
        # y_test = y_test
        # print(y_test[:,0])
        # print(np.round_(preds[:,i],2))
        fpr, tpr, tresholds = metrics.roc_curve(y_test[:,i],preds[:,i],drop_intermediate=True)

        conc = list(zip(fpr,tpr))
        # print(conc)
        # 0,1 is top - first x fpr then  y tpr
        for j,duo in enumerate(conc):
            #standard values is 0,1
            v1 = np.array([-0.6,1]) # (fpr, tpr) lowering the 1st value makes the model more strict in selecting ! -> reduces missclassification but also reduces 'hits' -> predict 0 more often
            v2 = np.array(conc[j])
            v3 = v1-v2
            mag = np.sqrt(v3.dot(v3))
            # if i == 2: print(v1,v2,mag)
            conc[j] = mag
            # conc[i] = list(np.array([0,1]) - conc[i] )
            # print(conc[i])
            # asd
        # best = [[0,1] - x for x in zip(conc[0],conc[1])]
        closest_treshold_top_left = tresholds[np.argmin(conc)]
        optimal_tresholds.append(closest_treshold_top_left)

        tn, fp, fn, tp = confusion_matrix((y_test[:,i]>optimal_tresholds[i]),(preds[:,i]>optimal_tresholds[i])).ravel()
        
        picked_fpr = fpr[np.argmin(conc)]
        picked_tpr = tpr[np.argmin(conc)]
        print(tn,fp,fn,tp)
        print("fpr = ",picked_fpr)
        print("tpr = ",picked_tpr)

        # print(ca.shape)
        # print(conc)
        # print( closest_top_left,tresholds[closest_top_left])
        
        # optimal_tresholds.append()
        # asd
        
        # optimal_tresholds.append(Find_Optimal_Cutoff(y_test[:,i],preds[:,i]))
        # print(np.round_(tresholds,2))

        # generating optimal tresholds (find where area under curve is maximum by multiplying fpr with reversed -> tpr)
        # fpr_tpr = np.multiply(fpr,tpr[::-1])
        # optimal_treshold = np.argmax(fpr_tpr)
        # optimal_treshold_value = tresholds[optimal_treshold]
        # optimal_tresholds.append(optimal_treshold_value)

        # print(fpr.shape,tpr.shape,tresholds.shape)
        auc = metrics.auc(fpr,tpr)
        import matplotlib.pyplot as plt
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title('ROC curve')
        plt.legend(loc='best')
        plt.show()
        aucs.append(auc)
        print(auc)
    print("mean auc",np.mean(aucs))
    print(np.round_(optimal_tresholds,7))
    # asd

 



    # '''
    # dataloader = MIR_DataLoader()
    # x_train, y_train = dataloader.load_data(train="test", model="with_postprocessing")

    # '''
    my_training_batch_generator = My_Custom_Generator(x_test, y_test, 1) #x_train file names
    for i in range(0,20):
        inp = my_training_batch_generator.__getitem__(i)[0]
        true = my_training_batch_generator.__getitem__(i)[1]

        # print(inp)
        print(true)




        # conv_net.compile(3e-4)
        # len_data = len(x_train)
        # conv_net.train_on_generator(my_training_batch_generator,epochs,len_data) 

        # conv_net.save("first test")
    

        preds  = conv_net.model.predict(inp).tolist() # thisd is predict
        # print("woof,",optimal_tresholds[i],preds[:,i])
        # print("\n\n\n")
        print(np.round_(preds,2))
        # print(optimal_tresholds[0])
        # asd
        # #unlist the list
        [preds] = preds
        # print(preds)
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
    # print(y_train[0:4])
    # '''
# '''