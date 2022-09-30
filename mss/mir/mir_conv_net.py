from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from mss.utils.dataloader import natural_keys, atof
from keras import Sequential
from tensorflow.keras import layers
import numpy as np
import keras
import os
import keras
import pickle
from pathlib import Path
from mss.mir.mir_data_load import MIR_DataLoader
from keras.callbacks import ModelCheckpoint, EarlyStopping
from sklearn import metrics
from tensorflow.keras import regularizers

from mss.mir.mir_data_load import MIR_DataLoader, My_Custom_Generator
from mss.utils.visualisation import visualize_loss, visualize_loss_auc, visualize_loss_val
from sklearn import metrics
from pathlib import Path
import tensorflow as tf


 
class ConvNet():

    def __init__(self, input_shape,
                    conv_filters: list,
                    conv_kernels: list,                   
                    conv_strides: list):  
        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self._num_conv_layers = len(conv_filters)

        self._name = ""
        self.weight_initializer = None # tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None )
        self.epoch_count = 1
        self.m_auc = 0
        self.patience = 50

        self.train_loss = []
        self.val_loss = []
        self.val_auc = []
        self.best_epoch = 0

        self._create_model()

    def save(self, save_folder=".",verbose=True):
        if verbose: print("saved:",save_folder)
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)
    
    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
        ]
        save_folder = "MIR_trained_models"/Path(save_folder)
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)
    
    def _save_weights(self, save_folder):
        save_folder = "MIR_trained_models"/Path(save_folder)
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def _create_folder_if_it_doesnt_exist(self, folder):
        folder = "MIR_trained_models"/Path(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)    

    @classmethod
    def load(cls, save_folder="."):
        temp = save_folder
        save_folder = "MIR_saved_models"/Path(save_folder); parameters = None;
        for i in range(2):
            try:  
                parameters_path = os.path.join(save_folder, "parameters.pkl")
                with open(parameters_path, "rb") as f:
                    parameters = pickle.load(f);break
            except FileNotFoundError as e: save_folder = "MIR_trained_models"/Path(temp)
        conv_net = ConvNet(*parameters)  # star for positional arguments!
        weights_path = os.path.join(save_folder, "weights.h5")
        conv_net.load_weights(weights_path)
        return conv_net

    def summary(self, save_image=False):
        import tensorflow 
        self.model.summary()
        tensorflow.keras.utils.plot_model(self.model, "model_conv.png", show_shapes=True)

    def compile(self, learning_rate=0.0001):
        
        optimizer = Adam(learning_rate=learning_rate)
        bce_loss = BinaryCrossentropy(from_logits=False)
        bce_loss = tf.keras.losses.Poisson(reduction="auto", name="poisson")
        self.model.compile(optimizer=optimizer, loss=bce_loss,metrics=[keras.metrics.BinaryAccuracy()])# ,self.sklearnAUC],run_eagerly=True) #self.custom_loss)dw  OR  ['accuracy'] for exact matching 
    
    def train_on_generator(self,model,batch_size,epochs):
        self.experiment_model = model
        dataloader = MIR_DataLoader()
        x_train, y_train = dataloader.load_data(dataset="train", model=model)
        x_test, y_test = dataloader.load_data(dataset="test", model=model)
        # x_train = x_train[:50]
        # y_train = y_train[:50]
        my_train_batch_generator = My_Custom_Generator(x_train, y_train, batch_size)
        my_test_batch_generator = My_Custom_Generator(x_test, y_test, batch_size)

        filepath="MIR_trained_models/mir_model_no_post_improve/weights-improvement-{epoch:02d}-{val_binary_accuracy:.2f}.h5"
        checkpoint = ModelCheckpoint(filepath, monitor='binary accuracy', verbose=1, save_best_only=True, mode='min')
        es = EarlyStopping(monitor='binary accuracy', mode='max', verbose=1, patience=5)        
     
        class_weights = MIR_DataLoader.get_train_class_weights()

        self.model.fit(my_train_batch_generator,
                        epochs = epochs, 
                        verbose = 1,
                        shuffle=True,                        
                        validation_data=my_test_batch_generator,
                        class_weight=class_weights,
                        callbacks=[CustomCallback(self) ]#, checkpoint ]
        )

    def _create_input(self,model):
        print(self.input_shape)
        model.add(layers.Input(shape=(self.input_shape),name="conv_net_input"))
        return model

    def _conv_block(self,model,i):
        #regularizer was 1e-5 for base and no post process -> regularizer was 1e-6 for with postprocessing due to validation loss viewings # MaYBE TRIE 1e-4 FOR NOPOSTPROCESS
        model.add(layers.Conv2D(self.conv_filters[i],self.conv_kernels[i],padding="same",kernel_initializer=self.weight_initializer,kernel_regularizer=regularizers.l1(1e-6)) ) # lower number (e8 < e6) means less regular #WAS 1e-8 -> used to be 1e-4
        # model.add(layers.Conv2D(self.conv_filters[i],self.conv_kernels[i],padding="same",kernel_initializer=self.weight_initializer) )
        # model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(self.conv_strides[i],self.conv_strides[i])))
        model.add(layers.Dropout(0.2)) #0.2
        return model
    
    def _dense_layer(self,model):
        model.add(layers.Flatten())
        model.add(layers.Dense(512,kernel_initializer=self.weight_initializer)) #higher value  (0.1 > 0.001) --> the more regularisation
        model.add(layers.Activation("relu"))
        # model.add(layers.BatchNormalization())
        model.add(layers.Dropout(0.5)) #0.3
        model.add(layers.Dense(11))
        return model
    
    def _output_layer(self,model):
        model.add(layers.Activation("sigmoid"))
        return model

    def _create_model(self):
        self.model = Sequential()

        # input block
        self.model = self._create_input(self.model)
       
        # conv blocks
        for i in range(self._num_conv_layers):
            self.model = self._conv_block(self.model,i)
        
        # dense layer
        self.model = self._dense_layer(self.model)
            
        # output layer
        self.model = self._output_layer(self.model)


class CustomCallback(keras.callbacks.Callback):
    def __init__(self,conv_net):
        self.conv_net = conv_net

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
       
        #calculate val_auc
        dataloader = MIR_DataLoader(verbose=False)  
        x_test, y_test = dataloader.load_data(dataset="test", model=self.conv_net.experiment_model) 
        my_test_batch_generator = My_Custom_Generator(x_test, y_test, 1)
        preds = self.conv_net.model.predict(my_test_batch_generator)       
        aucs = []
        for i in range(11):
            fpr, tpr, tresholds = metrics.roc_curve(y_test[:,i],preds[:,i])
            auc = metrics.auc(fpr,tpr)
            aucs.append(auc)        
 
        # save best model based on val auc
        if  np.mean(aucs) > self.conv_net.m_auc:
            self.conv_net.m_auc = np.mean(aucs)
            self.conv_net.patience=50
            print("improved auc")
            self.conv_net.save(self.conv_net._name)
            self.conv_net.best_epoch   =  self.conv_net.epoch_count
        else:
            self.conv_net.patience-=1
        
        # visualize stats    
        if len(self.conv_net.train_loss) == 0: # when starting new model; initialize values
            self.conv_net.train_loss    =  np.array([])
            self.conv_net.val_loss      =  np.array([])
            self.conv_net.val_auc       =  np.array([])
            self.conv_net.epoch_count   =  1
            self.conv_net.best_epoch    =  np.array([])

        self.conv_net.train_loss        = np.append(self.conv_net.train_loss,(logs["loss"]))
        self.conv_net.val_loss          = np.append(self.conv_net.val_loss,(logs["val_loss"]))
        self.conv_net.val_auc           = np.append(self.conv_net.val_auc,np.mean(aucs))
       
        np.save(f"MIR_trained_models/{self.conv_net._name}/train_loss.npy",self.conv_net.train_loss)
        np.save(f"MIR_trained_models/{self.conv_net._name}/val_loss.npy",self.conv_net.val_loss)
        np.save(f"MIR_trained_models/{self.conv_net._name}/val_auc.npy",self.conv_net.val_auc)
        np.save(f"MIR_trained_models/{self.conv_net._name}/epoch_count.npy",self.conv_net.epoch_count)
        np.save(f"MIR_trained_models/{self.conv_net._name}/best_epoch.npy",self.conv_net.best_epoch)
        
        try:
            visualize_loss_val( total_train_loss=self.conv_net.train_loss[2::], # start from epoch 2 for scale purposes
                            total_val_loss=self.conv_net.val_loss[2::],
                            smoothing=10,
                            model_name=f"/MIR/{self.conv_net._name}",
                            save=True)
            visualize_loss_auc( total_train_loss=self.conv_net.val_auc[2::],
                            total_val_loss=self.conv_net.val_auc[2::],
                            smoothing=10,
                            model_name=f"/MIR/{self.conv_net._name}",
                            save=True)
        except:
            print("will not visualize first 2 epoch for plot scaling purposes")

        with open (f"MIR_trained_models/{self.conv_net._name}/model_logs.txt","w") as f:
            f.write(f"current epoch:\t\t{self.conv_net.epoch_count}\n")
            f.write(f"best epoch:\t\t{self.conv_net.best_epoch}\n")
            f.write(f"best val_auc:\t\t{np.max(self.conv_net.val_auc)}\n")
            f.write(f"lowest train_loss:\t{np.min(self.conv_net.train_loss)}\n")
            f.write(f"lowest val_loss:\t{np.min(self.conv_net.val_loss)}\n")
            val_aucs = (list(enumerate(self.conv_net.val_auc,start=1)))
            for x,y in val_aucs:                
                y = str(np.round(y,4))+"\n"
                f.write(str(x)+"\t")
                f.write(str(y))
            f.close()

        #update epoch count
        self.conv_net.epoch_count+=1
       
        # verbose
        print(self.conv_net.m_auc)
        auc_stat = round(np.mean(aucs),4)
        print (f" - m_auc: {auc_stat}")


        if self.conv_net.patience <= 0:
            self.conv_net.model.stop_trainig = True #does not work
            # save last epoch (to continue from end point if we would want to)
            self.conv_net.save(self.conv_net._name+f"_last_epoch_{self.conv_net.epoch_count}")
            raise RuntimeError("Early stopping: Finished Training")
        else:
            # continues training
            pass
        






'''
class CustomCallback(keras.callbacks.Callback):
    def __init__(self,conv_net):
        self.conv_net = conv_net

    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())

        self.conv_net.train_loss.append(logs["loss"])
        self.conv_net.val_loss.append(logs["val_loss"])
        
        np.save(f"MIR_trained_models/{self.conv_net._name}/",self.conv_net.epoch_count)
        # print(self.conv_net.train_loss)

       
        if self.conv_net.epoch_count == 1:            
            self.conv_net._len_name = len(self.conv_net._name)

        # if model already exists
        try: self.conv_net.epoch_count = int(self.conv_net._name.split("_")[-1:][0])
        except: pass
        digit = self.conv_net._name.split("_")[-1:][0]
        if digit.isdigit():
            self.conv_net._name = self.conv_net._name[:-1].replace(digit,"")
            self.conv_net.epoch_count+=1

        self.conv_net._name = f"{self.conv_net._name[:self.conv_net._len_name]}_{self.conv_net.epoch_count}" # 100 epoch 

        # replace double stripe if model already exists       
        self.conv_net._name = self.conv_net._name.replace("__","_")

        self.conv_net.epoch_count+=1
        
        # new_name = self.conv_net._name+"_"+self.conv_net.epoch_count
        # conv_net = existing_model(model_name,new_name)

        dataloader = MIR_DataLoader(verbose=False)
        x_train, y_train = dataloader.load_data(dataset="train", model=self.conv_net.experiment_model)
        x_test, y_test = dataloader.load_data(dataset="test", model=self.conv_net.experiment_model) #"no_postprocessing"
        my_train_batch_generator = My_Custom_Generator(x_train, y_train, 1)
        my_test_batch_generator = My_Custom_Generator(x_test, y_test, 1)

        preds = self.conv_net.model.predict(my_test_batch_generator)

        from sklearn import metrics
        aucs = []
        for i in range(11):
            fpr, tpr, tresholds = metrics.roc_curve(y_test[:,i],preds[:,i])
            auc = metrics.auc(fpr,tpr)
            aucs.append(auc)
        

        if  np.mean(aucs) > self.conv_net.m_auc:
            self.conv_net.m_auc = np.mean(aucs)
            self.conv_net.patience=100
            print("improved auc")
            self.conv_net.save(self.conv_net._name)
        else:
            self.conv_net.patience-=1
            

        # self.conv_net.m_auc = np.mean(aucs)
        print(self.conv_net.m_auc)
        auc_stat = round(np.mean(aucs),4)
        print (f" - m_auc: {auc_stat}")
        
        self.conv_net.save(self.conv_net._name)
        # self.conv_net.save(self.conv_net._name+" auc "+str(auc_stat),verbose=False)

        if self.conv_net.patience <= 0:
            self.conv_net.model.stop_trainig = True #does not work
            asd
        else:
            pass
            # print(self.conv_net.patience)
        



        # self.conv_net.compile()
        lr =  None
        # print(self.conv_net.epoch_count)
        # '' ' 
        # if self.conv_net.epoch_count == 25+2 :  lr = 3e-4 ; K.set_value(self.conv_net.model.optimizer.learning_rate,lr)     #+2 because start at 2 after increment by 1 (start val)
        # if self.conv_net.epoch_count == 50+2 :  lr = 2e-4 ; K.set_value(self.conv_net.model.optimizer.learning_rate,lr)
        # if self.conv_net.epoch_count == 75+2 :  lr = 1e-4 ;K.set_value(self.conv_net.model.optimizer.learning_rate,lr)
        #  '' '    
    
        # print("reduced learn rate to ",lr)

        # print(keys)
        # print("End epoch {} of training; got log keys: {}".format(epoch, keys))


'''