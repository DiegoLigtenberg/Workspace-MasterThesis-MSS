from mss.mir.mir_conv_net import ConvNet
from mss.mir.mir_data_load import MIR_DataLoader, My_Custom_Generator
import numpy as np


LEARNING_RATE = 3e-4  #mnist 0.0001 
BATCH_SIZE = 16
EPOCHS = 15

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

def existing_model(model_name):
    conv_net = ConvNet.load(model_name)
    conv_net._name = model_name # new_name
    conv_net.summary()
    conv_net.compile(LEARNING_RATE)
    return conv_net

if __name__ == "__main__":
    model_name = "mir_model_auc" # 100 epoch
    
    
    conv_net = new_model(model_name)
    # conv_net = existing_model(model_name)
    conv_net.train_on_generator(model="base",batch_size=BATCH_SIZE,epochs=EPOCHS)
    

