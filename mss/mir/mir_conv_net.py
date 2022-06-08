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
        self._create_model()

    def save(self, save_folder="."):
        print("saved:",save_folder)
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
        self.model.compile(optimizer=optimizer, loss=bce_loss,metrics=[keras.metrics.BinaryAccuracy()]) #self.custom_loss)dw  OR  ['accuracy'] for exact matching 
    
    
    def train_on_generator(self,model,batch_size,epochs):
        dataloader = MIR_DataLoader()
        x_train, y_train = dataloader.load_data(train="train", model="base")
        x_test, y_test = dataloader.load_data("test", model=model)
        # x_train = x_train[:500]
        # y_train = y_train[:500]
        my_train_batch_generator = My_Custom_Generator(x_train, y_train, batch_size)
        my_test_batch_generator = My_Custom_Generator(x_test, y_test, batch_size)

        self.model.fit(my_train_batch_generator,
                        epochs = epochs, 
                        verbose = 1,
                        shuffle=True,                        
                        validation_data=my_test_batch_generator,
                        callbacks=[CustomCallback()]
        )

        # self.save(self._name)

    def _create_input(self,model):
        print(self.input_shape)
        model.add(layers.Input(shape=(self.input_shape),name="conv_net_input"))
        return model

    def _conv_block(self,model,i):
        model.add(layers.Conv2D(self.conv_filters[i],self.conv_kernels[i],padding="same",kernel_initializer=self.weight_initializer) )
        # model.add(layers.BatchNormalization())
        model.add(layers.Activation("relu"))
        model.add(layers.MaxPooling2D(pool_size=(self.conv_strides[i],self.conv_strides[i])))
        # model.add(layers.Dropout(0.5))
        return model
    
    def _dense_layer(self,model):
        model.add(layers.Flatten())
        model.add(layers.Dense(512,kernel_initializer=self.weight_initializer))
        model.add(layers.Activation("relu"))
        # model.add(layers.BatchNormalization())
        # model.add(layers.Dropout(0.2))
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
    def on_epoch_end(self, epoch, logs=None):
        keys = list(logs.keys())
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


def load_data():
    data = glob.glob(os.path.join("MIR_datasets/train_dataset/spectrogram_base", '*'),recursive=True)
    # data = sorted(data,key=os.path.getmtime)
    data.sort(key=natural_keys)
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

    x_train = []
    y_train = pd.read_csv("MIR_datasets/MIR_train_labels.csv").to_numpy()
    # y_train = np.delete(y_train, (0), axis=1) # make sure we dont save firrst column!

    for i,file in enumerate(data):
        x_train.append(file)

   


    x_train = np.array(x_train)
    y_train = np.array(y_train)
    return x_train,y_train

if __name__=="__main__":   
    
    x_train,y_train = load_data()
    x_train = x_train[0:8]
    y_train = y_train[0:8]
    
    # x_valid,yvalid = load_data() #valid data

    # print(x_train)
    # print(y_train)
    # asd
        
    conv_net = ConvNet(
        input_shape=(2048, 128, 1),
        conv_filters=(16, 32,  64, 128, 256,  256, 256), 
        conv_kernels=(3,   3,   3,   3,    2,   2,  2),
        conv_strides=(2,   2,   2,   2,    2,   2,  2), 
    )

    batch_size = 2
    epochs = 10
    # batch_size = 

    my_training_batch_generator = My_Custom_Generator(x_train, y_train, batch_size) #x_train file names

    # my_training_batch_generator_test = My_Custom_Generator(x_valid,y_valid)


    inp = my_training_batch_generator.__getitem__(0)[0]
    # print(inp)
    # print(inp.shape)
    # asd

    conv_net._name = ""
    conv_net.summary()
    conv_net.compile(3e-4)
    len_data = len(x_train)
    conv_net.train_on_generator(my_training_batch_generator,epochs) 

    conv_net.save("first test")
  

    preds  = conv_net.model.predict(inp).tolist() # thisd is predict
    preds = preds[0]
    # print(preds)
    # [preds] = preds # only take first element in batch
    # print(preds)
    from scipy.special import expit
    print(preds)
    # preds = expit(preds)
    preds = [round(num,2) for num in preds]
    print(preds)
    print(y_train[0:4])

