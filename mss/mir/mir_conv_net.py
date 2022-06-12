from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, BatchNormalization, Flatten, Dense, Reshape, Activation, Concatenate
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import BinaryCrossentropy
from mss.utils.dataloader import natural_keys, atof
from keras import Sequential
from tensorflow.keras import layers
import numpy as np
import tensorflow as tf
import keras
from torch import dropout
import pandas as pd
import keras
import glob
import os
import keras
from skimage.io import imread
from skimage.transform import resize
import pickle
from pathlib import Path
from mss.mir.mir_data_load import MIR_DataLoader
from keras.callbacks import ModelCheckpoint
from sklearn import metrics
from tensorflow.keras import regularizers
from keras import backend as K

def existing_model(model_name,new_name):
    conv_net = ConvNet.load(model_name)
    conv_net._name = new_name
    conv_net.summary()
    conv_net.compile(3e-4)
    return conv_net

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
        self.weight_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None )
        self.epoch_count = 1
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
        self.model.summary()

    def compile(self, learning_rate=0.0001):
        
        optimizer = Adam(learning_rate=learning_rate)
        bce_loss = BinaryCrossentropy(from_logits=False)
        self.model.compile(optimizer=optimizer, loss=bce_loss,metrics=[keras.metrics.BinaryAccuracy()])# ,self.sklearnAUC],run_eagerly=True) #self.custom_loss)dw  OR  ['accuracy'] for exact matching 
    
    def train_on_generator(self,model,batch_size,epochs):
        dataloader = MIR_DataLoader()
        x_train, y_train = dataloader.load_data(train="train", model=model)
        x_test, y_test = dataloader.load_data("test", model=model)
        # x_train = x_train[:50]
        # y_train = y_train[:50]
        my_train_batch_generator = My_Custom_Generator(x_train, y_train, batch_size)
        my_test_batch_generator = My_Custom_Generator(x_test, y_test, batch_size)

        # filepath="MIR_trained_models/mir_model_3/weights-improvement-{epoch:02d}-{val_binary_accuracy:.2f}.h5"
        # checkpoint = ModelCheckpoint(filepath, monitor='accuracy', verbose=1, save_best_only=True, mode='max')
        class_weights = {0:1.7880794701986755, 1:1.5254237288135593, 2:1.5055762081784387, 3:0.9225512528473804, 4:0.6666666666666666, 5:0.9507042253521126, 6:0.7390510948905109, 7:1.125, 8:1.2796208530805686, 9:1.2347560975609757, 10:0.6049290515309933}
        self.model.fit(my_train_batch_generator,
                        epochs = epochs, 
                        verbose = 1,
                        shuffle=True,                        
                        validation_data=my_test_batch_generator,
                        class_weight=class_weights,
                        callbacks=[CustomCallback(self) ]#, checkpoint ]
        )

        # self.save(self._name)

    def _create_input(self,model):
        print(self.input_shape)
        model.add(layers.Input(shape=(self.input_shape),name="conv_net_input"))
        return model

    def _conv_block(self,model,i):
        model.add(layers.Conv2D(self.conv_filters[i],self.conv_kernels[i],padding="same",kernel_initializer=self.weight_initializer,kernel_regularizer=regularizers.l1(1e-5)) )
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
        model.add(layers.Dropout(0.3)) #0.3
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
        x_train, y_train = dataloader.load_data(train="train", model="with_postprocessing")
        x_test, y_test = dataloader.load_data(train="test", model="with_postprocessing")
        my_train_batch_generator = My_Custom_Generator(x_train, y_train, 1)
        my_test_batch_generator = My_Custom_Generator(x_test, y_test, 1)

        preds = self.conv_net.model.predict(my_test_batch_generator)

        from sklearn import metrics
        aucs = []
        for i in range(11):
            fpr, tpr, tresholds = metrics.roc_curve(y_test[:,i],preds[:,i])
            auc = metrics.auc(fpr,tpr)
            aucs.append(auc)
        
        auc_stat = round(np.mean(aucs),4)
        print (f" - m_auc: {auc_stat}")

        
        self.conv_net.save(self.conv_net._name+" auc "+str(auc_stat),verbose=False)
        # self.conv_net.compile()
        lr =  None
        # print(self.conv_net.epoch_count)
        '''
        if self.conv_net.epoch_count == 25+2 :  lr = 3e-4 ; K.set_value(self.conv_net.model.optimizer.learning_rate,lr)     #+2 because start at 2 after increment by 1 (start val)
        if self.conv_net.epoch_count == 50+2 :  lr = 2e-4 ; K.set_value(self.conv_net.model.optimizer.learning_rate,lr)
        if self.conv_net.epoch_count == 75+2 :  lr = 1e-4 ;K.set_value(self.conv_net.model.optimizer.learning_rate,lr)
        '''    
    
        # print("reduced learn rate to ",lr)

        # print(keys)
        # print("End epoch {} of training; got log keys: {}".format(epoch, keys))


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





