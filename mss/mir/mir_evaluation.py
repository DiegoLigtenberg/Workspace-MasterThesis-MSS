
from mss.mir.mir_conv_net import ConvNet
from mss.mir.mir_data_load import MIR_DataLoader, My_Custom_Generator
from mss.mir.mir_train import existing_model

from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics  import f1_score
from sklearn.metrics import multilabel_confusion_matrix

import numpy as np

class Evaluate():
    pass


if __name__ == "__main__":
    model_name = "mir_model_dropout" # 100 epoch
    new_name = "mir_model_4"
    
    conv_net = existing_model(model_name,new_name)
    dataloader = MIR_DataLoader()
    x_train, y_train = dataloader.load_data(train="train", model="base")
    x_test, y_test = dataloader.load_data(train="test", model="base")
    # x_train = x_train[::5]
    # y_train = y_train[::5]

    my_train_batch_generator = My_Custom_Generator(x_train, y_train, 1)
    my_test_batch_generator = My_Custom_Generator(x_test, y_test, 1)

    # '''
    preds = conv_net.model.predict(my_test_batch_generator)
    
    # ca = multilabel_confusion_matrix((y_test>0.2),(preds>0.2))[1]
    # print(ca)
    # print(ca.shape)
    # '''

    

    # lrap = label_ranking_average_precision_score(y_test,preds)
    # print(lrap)

   
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
    aucs = []
    for i in range(11):
        # print(preds[:,0])
        # y_test = y_test
        # print(y_test[:,0])
        # print(np.round_(preds[:,i],2))
        fpr, tpr, tresholds = metrics.roc_curve(y_test[:,i],preds[:,i])
        # print(np.round_(tresholds,2))
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


    # '''
    # dataloader = MIR_DataLoader()
    # x_train, y_train = dataloader.load_data(train="test", model="with_postprocessing")

    '''
    my_training_batch_generator = My_Custom_Generator(x_test, y_test, 1) #x_train file names
    inp = my_training_batch_generator.__getitem__(50)[0]
    true = my_training_batch_generator.__getitem__(50)[1]

    # print(inp)
    print(true)




    # conv_net.compile(3e-4)
    # len_data = len(x_train)
    # conv_net.train_on_generator(my_training_batch_generator,epochs,len_data) 

    # conv_net.save("first test")
 

    preds  = conv_net.model.predict(inp).tolist() # thisd is predict
    preds = preds[0]
    # print(preds)
    # [preds] = preds # only take first element in batch
    # print(preds)
    # from scipy.special import expit
    # print(preds)
    # preds = expit(preds)
    preds = [round(num,2) for num in preds]
    print(preds)
    # print(y_train[0:4])
    '''
# '''