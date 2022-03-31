from encodings.utf_8 import encode
from enum import auto
from re import M
from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Conv2D, Conv2DTranspose, ReLU, BatchNormalization, Flatten, Dense, Reshape, Activation, Concatenate
from tensorflow.keras import backend as K
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.losses import MeanSquaredError
import numpy as np
import os
import pickle

import tensorflow as tf


class VariationalAutoEncoder():
    """
    Variational Autoencoder represents a Deep Convolutional variational autoencoder architecture with
    mirrored encoder and decoder components.
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
        '''private and protected does not exist in python, so this is just convention, but not neccesary!'''
        # _variables or _functions are protected variables/functions and can only be used in subclasses, but can be overwritten by subclasses
        # __variables or __functions are private classes and can not EASILY be used in other classes/subclasses because the name does not show up on top!
        self._build()

    def save(self, save_folder="."):
        self._create_folder_if_it_doesnt_exist(save_folder)
        self._save_parameters(save_folder)
        self._save_weights(save_folder)

    def reconstruct(self, images):
        latent_representations = self.encoder.predict(images)
        reconstructed_images = self.decoder.predict(latent_representations)
        return reconstructed_images, latent_representations

    @classmethod
    def load(cls, save_folder="."):
        parameters_path = os.path.join(save_folder, "parameters.pkl")
        with open(parameters_path, "rb") as f:
            parameters = pickle.load(f)
        variational_auto_encoder = VariationalAutoEncoder(*parameters)  # star for positional arguments!
        weights_path = os.path.join(save_folder, "weights.h5")
        variational_auto_encoder.load_weights(weights_path)
        return variational_auto_encoder

    def load_weights(self, weights_path):
        self.model.load_weights(weights_path)

    def _create_folder_if_it_doesnt_exist(self, folder):
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
        save_path = os.path.join(save_folder, "parameters.pkl")
        with open(save_path, "wb") as f:
            pickle.dump(parameters, f)

    def _save_weights(self, save_folder):
        save_path = os.path.join(save_folder, "weights.h5")
        self.model.save_weights(save_path)

    def summary(self, save_image=False):
        # self.encoder.summary()
        # self.decoder.summary()
        self.model.summary()
        # if save_image:
        #     keras.utils.plot_model(self.encoder, "encoder_model.png", show_shapes=True)
        #     keras.utils.plot_model(self.decoder, "decoder_model.png", show_shapes=True)

    def compile(self, learning_rate=0.0001):
        optimizer = Adam(learning_rate=learning_rate)
        mse_loss = MeanSquaredError()
        self.model.compile(optimizer=optimizer, loss=mse_loss)

    def train(self, x_train, y_train, batch_size, num_epoch):
        # since we try to reconstruct the input, the output y_train is basically also x_train
        # y_train=x_train
        self.model.fit(x_train, y_train,
                       batch_size=batch_size,
                       epochs=num_epoch,
                       shuffle=True)

    def _build(self):
        # self._build_encoder()
        # self._build_decoder()
        self._build_autoencoder()

    def _add_encoder_input(self):
        '''returns Input Object - Keras Input Layer object'''
        return Input(shape=self.input_shape, name="encoder_input")  # returns the input shape of your data

    def _add_conv_layers(self, encoder_input):
        '''Creates all convolutional blocks in the encoder'''
        model = encoder_input
        self.encoder_list = []
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
            # (int) amount of kernels we use -> output dimensionality of this conv layer
            filters=self.conv_filters[layer_index],
            # filter size over input (4 x 4) -> can also be rectengular
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            # keeps dimensionality same -> adds 0's outside the "image" to make w/e stride u pick work
            padding="same",
            name=f"encoder_conv_layer{layer_number}"
        )

        '''adding Conv, Relu, and Batch normalisation to each layer -> x is now the model'''
        # model = model (Input) + Conv layers
        # add the convolutional layers to whatever x was
        model = conv_layer(model)

        model = ReLU(name=f"encoder_relu_{layer_number}")(model)
        model = BatchNormalization(name=f"encoder_bn_{layer_number}")(model)
        # print("shape",model)
        self.encoder_list.append(model)

        return model

    def _add_bottle_neck(self, model):
        '''Flatten data and add bottleneck ( Dense Layer ). '''
        self._shape_before_bottleneck = K.int_shape(model)[
                                        1:]  # [2, 7 ,7 , 32] # 4 dimensional array ( batch size x width x height x channels )
        model = Flatten()(model)
        model = Dense(self.latent_space_dim, name="encoder_output")(
            model)  # dimensionality of latent space -> outputshape
        # each output layer in dense layer is value between 0 and 1, if the value is highest -> then we pick that output
        # for duo classification you have 1 layer between 0 and 1 , if the value is > 0.5 then we pick that output

        return model

    def _add_decoder_input(self):
        return Input(shape=self.latent_space_dim, name="decoder_input")

    def _add_dense_layer(self, decoder_input):
        # product of neurons from previous conv output in dense layer
        num_neurons = np.prod(self._shape_before_bottleneck)
        dense_layer = Dense(num_neurons, name="decoder_dense")(decoder_input)
        return dense_layer

    def _add_reshape_layer(self, dense_layer):
        reshape_layer = Reshape(self._shape_before_bottleneck)(dense_layer)
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
            filters=self.conv_filters[layer_index ],
            kernel_size=self.conv_kernels[layer_index],
            strides=self.conv_strides[layer_index],
            padding="same",
            name=f"decoder_conv_transpose_layer_{layer_number}"
        )

        x = conv_transpose_layer(x)
        if layer_index == 1:
            pass
        x = ReLU(name=f"decoder_relu_{layer_number}")(x)
        x = BatchNormalization(name=f"decoder_bn_{layer_number}")(x)
        x = Concatenate(axis=3)([self.encoder_list[layer_index - 1], x])  # U-net skip connections        
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
            name=f"decoder_conv_transpose_layer_{self._num_conv_layers}"
        )
        x = conv_transpose_layer(x)
        output_layer = Activation("tanh", name="tanh_output_layer")(x) # pogchamp rob - sigmoid -> tanh because normalisation
        return output_layer

    def _build_autoencoder(self):
        ####### encoder
        encoder_input = self._add_encoder_input()
        conv_layers = self._add_conv_layers(encoder_input)
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

        self.model = Model(encoder_input, decoder_output, name="Autoencoder")


if __name__ == "__main__":
    variational_auto_encoder = VariationalAutoEncoder(
        input_shape=(28, 28, 1),
        conv_filters=(32, 64, 64, 64),
        conv_kernels=(3, 3, 3, 3),
        # stride of 2 is downsampling the data -> halving it!
        conv_strides=(1, 2, 2, 1),
        latent_space_dim=2)

    variational_auto_encoder.summary(save_image=False)
