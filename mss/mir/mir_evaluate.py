
from mss.mir.mir_conv_net import ConvNet
from mss.mir.mir_data_load import MIR_DataLoader, My_Custom_Generator
from mss.mir.mir_train import existing_model
from mss.mir.mir_generate_labels import LabelGenerator

from sklearn.metrics import confusion_matrix, roc_auc_score, roc_curve
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics  import f1_score
from sklearn.metrics import multilabel_confusion_matrix

from sklearn import metrics
import numpy as np
import matplotlib.pyplot as plt
import os

class Conv_Statistics():
    def __init__(self,verbose=False):
        self.verbose = verbose
        

    def calculate_lrap(self,y_test,preds):
        lrap = label_ranking_average_precision_score(y_test,preds)
        return lrap
    
    def calculate_m_auc(self,y_test,preds,model_name,save_ROC):
        aucs = []
        optimal_tresholds = []
        instrument_names =  ["cel","cla","flu","gac", "gel","org","pia","sax","tru", "vio", "voi"]
   
        for i in range(11):
            instrument = instrument_names[i]
            # calculates fpr, tpr for each column (instrument) in y_test (766 x 11); taking ith column corresponding to the instrument
            fpr, tpr, tresholds = metrics.roc_curve(y_test[:,i],preds[:,i],drop_intermediate=True)
            auc = metrics.auc(fpr,tpr)
            aucs.append(auc)

            self._find_optimal_cutoff(y_test,preds,fpr,tpr,tresholds,optimal_tresholds,i)
            self._visualize_roc(fpr,tpr,auc,instrument,model_name,save_ROC,i)
        
        self.mean_auc = np.mean(aucs)
        self.optimal_tresholds = optimal_tresholds
        return self.mean_auc,optimal_tresholds
    
    def _find_optimal_cutoff(self,y_test,preds,fpr,tpr,tresholds,optimal_tresholds,i):
        '''
        finds cutoff corresponding to a specified area in ROC Curve
        i.e. closest to top left (maximizing area)'''
        conc = list(zip(fpr,tpr))
        # 0,1 is top - first x fpr then y tpr
        for j,duo in enumerate(conc):
            #standard values is 0,1
            v1 = np.array([-1,1])  # (fpr, tpr) lowering the 1st value makes the model more strict in selecting ! 
            v2 = np.array(duo)
            v3 = v1-v2
            mag = np.sqrt(v3.dot(v3))
            conc[j] = mag
        closest_treshold_top_left = tresholds[np.argmin(conc)]
        optimal_tresholds.append(closest_treshold_top_left)
        tn, fp, fn, tp = confusion_matrix((y_test[:,i]>optimal_tresholds[i]),(preds[:,i]>optimal_tresholds[i]),labels=[0,1]).ravel()        
        picked_fpr = fpr[np.argmin(conc)]
        picked_tpr = tpr[np.argmin(conc)]
        # if self.verbose: print("tn",tn,"fp",fp,"fn",fn,"tp",tp);print("fpr = ",picked_fpr);print("tpr = ",picked_tpr)

    def _visualize_roc(self,fpr,tpr,auc,instrument,model_name,save_ROC,i):
        folder = f"visualisation/MIR/{model_name}/ROC_Curves/"
        if not os.path.exists(folder):
            os.makedirs(folder)
            
        plt.figure(1)
        plt.plot([0, 1], [0, 1], 'k--')
        plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
        plt.xlabel('False positive rate')
        plt.ylabel('True positive rate')
        plt.title(f'{instrument} ROC curve')
        plt.legend(loc='best')
        print(instrument,auc)
        # if save_ROC: plt.savefig(f"{folder}ROC_{instrument}.png")
        write_param = "w" if i == 0 else "a"
        with open (f"{folder}_instrument_aucs.txt",write_param) as f:
            text = instrument+" "+str(np.round(auc,4))+"\n"
            f.writelines(text)

        # plt.show()
        plt.close()

class Evaluate():
    def __init__(self,models,model):
        self.models = models
        self.model = model
        self.dataloader =  MIR_DataLoader()
        self.statistics = Conv_Statistics(verbose=True)

        for model_name in self.models:
            self.evaluate_data(model_name,model)
        
            self.__visualize_predictions(model_name,model)

    def evaluate_data(self,model_name,model):
        #evaluates all models passed in
        my_val_batch_generator, my_test_batch_generator, y_val, y_test = self._load_eval_data(model_name,model)
        preds = self._predict(my_test_batch_generator,model_name)
        lrap = self.statistics.calculate_lrap(y_test,preds)
        m_auc,self.optimal_tresholds = self.statistics.calculate_m_auc(y_test,preds,model_name,save_ROC=True)
        print("\nm_auc =\t",m_auc)
        print("lrap =\t",lrap)
        

    def _load_eval_data(self,model_name,model):
        x_val, y_val = self.dataloader.load_data(dataset="val", model=model)
        x_test, y_test = self.dataloader.load_data(dataset="test", model=model)     
        my_val_batch_generator = My_Custom_Generator(x_val, y_val, 1)
        my_test_batch_generator = My_Custom_Generator(x_test, y_test, 1)
        return my_val_batch_generator,my_test_batch_generator, y_val, y_test
    
    def __visualize_predictions(self,model_name,model):
        print("optimal tresholds = ",self.optimal_tresholds)
        my_test_batch_generator = self._load_eval_data(model_name,model)[1]
        for i in range(0,1):
            inp = my_test_batch_generator.__getitem__(i)[0]
            true = my_test_batch_generator.__getitem__(i)[1]
      
            try:
                # print(inp)
                q = true
                [q] = q
            except:
                print("last prediction")
                break
            if q[6] == 1: #this q indicates which individual instrument we want to see results for
                print(i,LabelGenerator.i2f()[i])
                print(q)
                preds  = self.conv_net.model.predict(inp).tolist() 
                [preds] = preds
                # print(np.round(preds,2))
                preds = np.array(preds)
                for j in range(11):
                    preds[j] = (preds[j]> self.optimal_tresholds[j]) 
                print(preds,"\n")
        
    def _predict(self,batch_generator,model_name):
        self.conv_net = existing_model(model_name)
        preds = self.conv_net.model.predict(batch_generator)
        return preds

        
evaluate = Evaluate(["base"],model="base")

