
from torch import float16
from mss.mir.mir_conv_net import ConvNet
from mss.mir.mir_data_load import MIR_DataLoader
from sklearn.metrics import roc_auc_score, roc_curve
from sklearn.metrics import label_ranking_loss
from sklearn.metrics import label_ranking_average_precision_score
from sklearn.metrics  import f1_score
from sklearn.metrics import multilabel_confusion_matrix

LEARNING_RATE = 3e-4  #mnist 0.0001 
BATCH_SIZE = 16
EPOCHS = 7

def new_model(model_name):
    conv_net = ConvNet(
        input_shape=(2048, 128, 1),
        conv_filters=(16, 32,  64, 128, 256,  256, 256), 
        conv_kernels=(3,   3,   3,   3,    2,   2,  2),
        conv_strides=(2,   2,   2,   2,    2,   2,  2), 
    )
    conv_net._name = model_name
    conv_net.summary()
    conv_net.compile(LEARNING_RATE)
    return conv_net 

def existing_model(model_name,new_name):
    conv_net = ConvNet.load(model_name)
    conv_net._name = new_name
    conv_net.summary()
    conv_net.compile(LEARNING_RATE)
    return conv_net
    
import keras
import numpy as np
class My_Custom_Generator(keras.utils.all_utils.Sequence):
    '''
    input:  (X_file names, Y labels)
    output: (X_file.npy, Y_labels)
    '''

    def __init__(self,image_file_names,labels,batch_size):
        self.image_file_names = image_file_names
        self.labels = labels
        self.batch_size = batch_size

    def __len__(self):
        return (np.ceil(len(self.image_file_names) / float(self.batch_size))).astype(np.int)
    
    def __getitem__(self,idx):
        batch_x = self.image_file_names[idx * self.batch_size : (idx+1) * self.batch_size]
        batch_y = self.labels[idx*self.batch_size : (idx+1) * self.batch_size]
        # batch = []
        # batch = np.array(batch)
        # for i,file_name in enumerate(batch_x):
        #     row_to_append = np.array([np.load(file_name).astype('float32') ,batch_y[i].astype('float32')]  )
            
        #     batch = np.append(batch,row_to_append,0)
        # batch = np.array(batch)
        # print(batch.shape)
        # asd
        return np.array([
            (np.load(str(file_name)))
               for file_name in batch_x]), np.array(batch_y)

if __name__ == "__main__":
    model_name = "mir_model_2"
    new_name = "mir_model_3"
    
    # conv_net = new_model(model_name)
    conv_net = existing_model(model_name,new_name)
    # conv_net.train_on_generator(model="base",batch_size=BATCH_SIZE,epochs=EPOCHS)


    dataloader = MIR_DataLoader()
    x_train, y_train = dataloader.load_data(train="train", model="base")
    x_test, y_test = dataloader.load_data(train="test", model="with_postprocessing")

    my_train_batch_generator = My_Custom_Generator(x_train, y_train, 16)
    my_test_batch_generator = My_Custom_Generator(x_test, y_test, 16)

    '''
    preds = conv_net.model.predict(my_test_batch_generator)
    
    ca = multilabel_confusion_matrix((y_test>0.2),(preds>0.2))[1]
    print(ca)
    print(ca.shape)
    '''
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

    # fpr, tpr, tresholds = roc_curve(y_test,preds)

    # auc = roc_auc_score(fpr,tpr)
    # import matplotlib.pyplot as plt
    # plt.figure(1)
    # plt.plot([0, 1], [0, 1], 'k--')
    # plt.plot(fpr, tpr, label='Keras (area = {:.3f})'.format(auc))
    # plt.xlabel('False positive rate')
    # plt.ylabel('True positive rate')
    # plt.title('ROC curve')
    # plt.legend(loc='best')
    # plt.show()

    # print(auc)


    '''
    dataloader = MIR_DataLoader()
    x_train, y_train = dataloader.load_data(train="test", model="with_postprocessing")

    my_training_batch_generator = My_Custom_Generator(x_train, y_train, 1) #x_train file names
    inp = my_training_batch_generator.__getitem__(80)[0]
    true = my_training_batch_generator.__getitem__(80)[1]

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
    from scipy.special import expit
    # print(preds)
    # preds = expit(preds)
    preds = [round(num,2) for num in preds]
    print(preds)
    # print(y_train[0:4])
'''