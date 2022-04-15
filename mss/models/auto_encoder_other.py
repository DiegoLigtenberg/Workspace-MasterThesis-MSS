from encodings.utf_8 import encode
from enum import auto
from re import M
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, BatchNormalization, Flatten, Dense, Reshape, Activation, Concatenate, Dropout, Multiply
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
from tensorflow.keras import regularizers
import numpy as np
import os
import pickle

from mss.utils.dataloader import DataLoader
from mss.utils.visualisation import visualize_loss
from tensorflow.keras.utils import Progbar

import tensorflow as tf
# import keras as keras
from pathlib import Path

import random

class AutoEncoder():
    """
     Autoencoder represents a Deep Convolutional  autoencoder architecture with
    mirrored encoder and decoder components.
    - skip connections are added compared to Vanilla auto encoder -> changed model structure by building model only after encode+decode st it now concatenates like (unet)
    """

    def __init__(self, input_shape,  # width x height x nr channels (rgb)    -> or [28 x 28 x 1] for black/white
                 # list of filter sizes for each layer   [2,4,8 ] 1st layer 2x2, 2nd layer 4x4 etc..
                 conv_filters: list,
                 # list of kernel sizes for each layer   [3,5,3 ] 1st layer 3x3, 2nd layer 5x5 etc..
                 conv_kernels: list,
                 # list of stride sizes for each layer   [1,2,2 ] 1st layer 1x1, 2nd layer 2x2 etc..
                 conv_strides: list,
                 latent_space_dim):  # int #number of dimensions of bottleneck

        self.input_shape = input_shape
        self.conv_filters = conv_filters
        self.conv_kernels = conv_kernels
        self.conv_strides = conv_strides
        self.latent_space_dim = latent_space_dim

        self.encoder = None
        self.decoder = None
        self.model = None

        self._num_conv_layers = len(conv_filters)  # dimension of amnt kernels
        self._shape_before_bottleneck = None
        self._model_input = None
        # tf.compat.v1.disable_eager_execution() # works for random seed
        tf.random.set_seed(1)
        # self.weight_initializer = tf.initializers. TruncatedNormal(mean=0., stddev=1/1024)
        self.weight_initializer = tf.keras.initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None )
        self.regularizer =  regularizers.l2(1e-3)
        self.name = ""

        '''private and protected does not exist in python, so this is just convention, but not neccesary!'''
        # _variables or _functions are protected variables/functions and can only be used in subclasses, but can be overwritten by subclasses
        # __variables or __functions are private classes and can not EASILY be used in other classes/subclasses because the name does not show up on top!
        self._build()

    def save(self, save_folder="."):
        print("saved:",save_folder)
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, save_folder="."):
        save_folder = "trained_models"/Path(save_folder)
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        variational_auto_encoder = AutoEncoder(*parameters)  # star for positional arguments!
        weights_path = os.path.join(save_folder, "weights.h5")
        variational_auto_encoder.load_weights(weights_path)
        return variational_auto_encoder

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def _create_folder_if_it_doesnt_exist(self, folder):
        folder = "trained_models"/Path(folder)
        if not os.path.exists(folder):
            os.makedirs(folder)

    def _save_parameters(self, save_folder):
        parameters = [
            self.input_shape,
            self.conv_filters,
            self.conv_kernels,
            self.conv_strides,
            self.latent_space_dim
        ]
        save_folder = "trained_models"/Path(save_folder)
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_folder = "trained_models"/Path(save_folder)
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def summary(self, save_image=False):
        # self.encoder.summary()
        # self.decoder.summary()
        import tensorflow
        self.model.summary()
        if save_image:
            tensorflow.keras.utils.plot_model(self.model, "model_multiply.png", show_shapes=True)
        #     keras.utils.plot_model(self.decoder, "decoder_model.png", show_shapes=True)

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)

    def train(self,x_train,y_train,batch_size,num_epoch):
        # since we try to reconstruct the input, the output y_train is basically also x_train
        # y_train=x_train
        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=num_epoch,
                       shuffle=True)

    def train_on_batch(self, batch_size, num_epoch):
        metrics_names = ['train loss','mean loss','val_loss','mean val_loss'] 
        self.dataloader = DataLoader(batch_size=batch_size,num_epoch=num_epoch)

        self.loss = []
        meanloss = 0

        self.val_loss_m = []
        meanloss_val = 0
        val_loss2 = 0

        total_train_loss = []
        total_val_loss = []
        try:            
            total_train_loss = list(np.load(f"visualisation/{self.name}/total_train_loss.npy"))
            total_val_loss = list(np.load(f"visualisation/{self.name}/total_val_loss.npy"))
            print("loaded loss files")
        except:
            print("no file of previous loss yet")

        for epoch_nr in range(0, num_epoch):
            pb_i = Progbar(self.dataloader.len_train_data, stateful_metrics=metrics_names)
            print("\nepoch {}/{}".format(epoch_nr+1,num_epoch))
            for batch_nr in range(self.dataloader.nr_batches):
                try:
                    x_train, y_train = self.dataloader.load_data(batch_nr=batch_nr)
                    loss = self.model.train_on_batch(x_train, y_train) 
                    loss2 = float(str(loss)) #[0:9])
                    self.loss.append(loss)  
                    
                    meanloss = np.mean(self.loss) 
                    meanloss = float(str(meanloss)) #[0:9])
                    if  batch_nr % 6 == 0 :
                        x_val, y_val = self.dataloader.load_val(batch_nr=batch_nr)
                        # val_loss = self.model.train_on_batch(x_val, y_val) 
                        # print("val loss 1",val_loss)
                        y_pred = self.model.predict(x_val)
                        y_pred = tf.convert_to_tensor(y_pred,dtype=tf.float32)
                        y_val = tf.cast(y_val, y_pred.dtype)
                        val_loss = K.mean(tf.math.squared_difference(y_pred, y_val), axis=-1)
                        # val_loss = val_loss.eval(session=tf.compat.v1.Session()) # if eager execution
                        val_loss = np.mean(val_loss.numpy())
                        # print("val loss 2",val_loss)
                        val_loss2 = float(str(val_loss)) #[0::])
                        self.val_loss_m.append(val_loss)                      
                        meanloss_val = np.mean(self.val_loss_m)
                        meanloss_val = float(str(meanloss_val)) #[0:9])
                    if batch_nr %100 == 0:   
                        # total_train_loss.append(loss)
                        # total_val_loss.append(val_loss)  
                        pass 
                    
                    values=[('train loss',loss2),("mean loss",meanloss),("val_loss",val_loss2),("mean val_loss",meanloss_val)]  # add comma after last ) to add another metric!        
                    pb_i.add(batch_size, values=values)

                except:
                    pass
            total_train_loss.append(meanloss)
            total_val_loss.append(meanloss_val)  
            self.dataloader.shuffle_data()
            self.dataloader.reset_counter() # makes it work after last epoch            
            visualize_loss(total_train_loss[-(len(total_train_loss))-1::],total_val_loss[-(len(total_train_loss))-1::],save=True,model_name=self.name) # adding first is buggy
            
            if epoch_nr%10 == 0:
                self.save(f"{self.name}-{epoch_nr}-{round(meanloss,5)}")
                pass
            self.loss = []
            self.val_loss_m = []


    def _build(self):
        # self._build_encoder()
        # self._build_decoder()
        self._build_autoencoder()

    def _add_encoder_input(self):
        '''returns Input Object - Keras Input Layer object'''        
        self.encoder_list = []
        inp = Input(shape=self.input_shape, name="encoder_input")        
        self.encoder_list.append(inp)
        return  inp # returns the input shape of your data

    def _add_conv_layers(self, encoder_input):
        '''Creates all convolutional blocks in the encoder'''
        model = encoder_input
        # layer_index tells us at which layer we pass in the specific conv layer
        for layer_index in range(self._num_conv_layers):
            # will now be a graph of layers
            model = self._add_conv_layer(layer_index, model)
        return model

    def _add_conv_layer(self, layer_index, model):
        '''adds a conv layer to the total neural network network that started with only Input()'''
        '''
        Adds a convolutional block to a graph of layers, consisting of 
        conv 2d + 
        Relu activation +
        Batch normalization   
        '''
        layer_number = layer_index + 1
        conv_layer = Conv2D(
            # (int) amount of kernels we use -> output dimensionality of this conv layer -> how many filters we use
            filters=self.conv_filters[layer_index],
            # filter size over input (4 x 4) -> can also be rectengular
            kernel_size=(self.conv_kernels[layer_index],self.conv_kernels[layer_index]),
            strides=self.conv_strides[layer_index],
            # keeps dimensionality same -> adds 0's outside the "image" to make w/e stride u pick work
            padding="same",
            name=f"encoder_conv_layer{layer_number}",
            kernel_initializer=self.weight_initializer
        )

        '''adding Conv, Relu, and Batch normalisation to each layer -> x is now the model'''
        # model = model (Input) + Conv layers
        # add the convolutional layers to whatever x was
        model = conv_layer(model)

        model = ReLU(name=f"encoder_relu_{layer_number}")(model)
        model = BatchNormalization(name=f"encoder_bn_{layer_number}")(model)        
        if layer_index < 3:
            model = Dropout(0.3)(model)
            pass
        # print("shape",model)
        self.encoder_list.append(model)

        return model

    def _add_bottle_neck(self, model):
        '''Flatten data and add bottleneck ( Dense Layer ). '''
        self._shape_before_bottleneck = K.int_shape(model)[
                                        1:]  # [2, 7 ,7 , 32] # 4 dimensional array ( batch size x width x height x channels )
        model = Flatten()(model)
        model = Dense(self.latent_space_dim, name="encoder_output",kernel_regularizer=self.regularizer)(model)  # dimensionality of latent space -> outputshape
        # each output layer in dense layer is value between 0 and 1, if the value is highest -> then we pick that output
        # for duo classification you have 1 layer between 0 and 1 , if the value is > 0.5 then we pick that output

        return model

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        # product of neurons from previous conv output in dense layer
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = Dense(num_neurons, name="decoder_dense",kernel_regularizer=self.regularizer)(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer)
        reshape_layer = Concatenate(axis=3)([self.encoder_list[len(self.encoder_list)-1], reshape_layer])  # U-net skip connections      
        return reshape_layer

    def _add_conv_transpose_layers(self, x):
        '''add convolutional transpose blocks -> conv2d -> relu -> batch normalisation'''
        # loop through all the conv layers in reverse order and stop at the first layer
        # [0, 1 , 2 ] -> [ 2, 1 ] remove first value
        for layer_index in reversed(range(1, self._num_conv_layers)):
            x = self._add_conv_transpose_layer(layer_index, x)
        return x

    def _add_conv_transpose_layer(self, layer_index, x):
        layer_number = self._num_conv_layers - layer_index
        conv_transpose_layer = Conv2DTranspose(
            filters=self.conv_filters[layer_index-1 ],
            kernel_size=(self.conv_kernels[layer_index-1],self.conv_kernels[layer_index-1]),
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_number}",
            kernel_initializer=self.weight_initializer,
            kernel_regularizer=self.regularizer
        )

        x = conv_transpose_layer(x)
        if layer_index == 1:
            pass
        x = ReLU(name=f"decoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_number}")(x)
        x = Concatenate(axis=3)([self.encoder_list[layer_index ], x])  # U-net skip connections        
        return x

    def _add_decoder_output(self, x):
        conv_transpose_layer = Conv2DTranspose(
            # [ 24 x 24 x 1] # number of channels = 1 thus filtersd is 1
            filters=1,
            # first that we skipped on _add_conv_transpose_layer
            kernel_size=self.conv_kernels[0],
            # first that we skipped on _add_conv_transpose_layer
            strides=self.conv_strides[0],
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}",
            kernel_initializer=self.weight_initializer
        )
        x = conv_transpose_layer(x) 
        # x = Concatenate(axis=3)([self.encoder_list[0], x])  # U-net skip connections
        
        conv_transpose_layer = Conv2DTranspose(
            # [ 24 x 24 x 1] # number of channels = 1 thus filtersd is 1
            filters=1,
            # first that we skipped on _add_conv_transpose_layer
            kernel_size=(1,1),
            # first that we skipped on _add_conv_transpose_layer
            strides=(1,1),
            padding="same",
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers+1}",
            kernel_initializer=self.weight_initializer,
            kernel_regularizer=self.regularizer
        )
        # x = conv_transpose_layer(x)
        # output_layer = Activation("sigmoid", name="softmax_output_layer")(x)
        # output_layer = Multiply(name="multiply")([x,self.encoder_input])
        output_layer = Activation("tanh", name="tanh_output_layer")(x) # pogchamp rob - sigmoid -> tanh because normalisation
        # output_layer = Multiply(name="multiply")([output_layer,self.encoder_input])
      
        return output_layer

    def _build_autoencoder(self):
        ####### encoder
        self.encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(self.encoder_input)
        bottleneck = self._add_bottle_neck(conv_layers)

        # self._model_input = encoder_input
        # self.encoder = Model(encoder_input, bottleneck, name="encoder")

        ####### decoder
        # decoder_input = self.encoder #self._add_decoder_input()
        dense_layer = self._add_dense_layer(bottleneck)
        reshape_layer = self._add_reshape_layer(dense_layer)

        conv_transpose_layers = self._add_conv_transpose_layers(reshape_layer)
        decoder_output = self._add_decoder_output(conv_transpose_layers)
        # print("input",decoder_input)
        # self.decoder = Model(decoder_input, decoder_output, name="decoder") 
        # print(decoder_output)

        self.model = Model(self.encoder_input, decoder_output, name="Autoencoder")

def main():
    auto_encoder = AutoEncoder(
    input_shape=(2048, 128, 1),
    conv_filters=(16, 32, 64, 128, 256, 512),
    conv_kernels=(3, 3, 3, 3, 3, 3),
    # stride of 2 is downsampling the data -> halving it!
    conv_strides=(2, 2, 2, 2, 2, 2),
    latent_space_dim=128)

    auto_encoder.summary(save_image=True)

if __name__ == "__main__":
    main()
