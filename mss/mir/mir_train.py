from mss.mir.mir_conv_net import ConvNet
import numpy as np

LEARNING_RATE = 0.00025 #mnist 0.0001  # 0.00025    #finetune = 0.000075
BATCH_SIZE = 16
EPOCHS = 500

def new_model(model_name):
    conv_net = ConvNet(
        input_shape=(2048, 128, 1),
        conv_filters=(32, 32,  64, 128, 256,  256, 256), 
        conv_kernels=(7,   5,   3,   3,    2,   2,  2),
        conv_strides=(2,   2,   2,   2,    2,   2,  2), 
    )
    conv_net._name = model_name
    conv_net.summary()
    conv_net.compile(LEARNING_RATE)
    return conv_net 

def existing_model(model_name):
    conv_net = ConvNet.load(model_name)
    conv_net._name = model_name # new_name
    # conv_net.summary()
    conv_net.compile(LEARNING_RATE)

    try:
        conv_net.best_epoch    =  np.load(f"MIR_trained_models/{conv_net._name}/best_epoch.npy")
    
        conv_net.train_loss    =  np.load(f"MIR_trained_models/{conv_net._name}/train_loss.npy")[:conv_net.best_epoch]
        conv_net.val_loss      =  np.load(f"MIR_trained_models/{conv_net._name}/val_loss.npy")[:conv_net.best_epoch]
        conv_net.val_auc       =  np.load(f"MIR_trained_models/{conv_net._name}/val_auc.npy")[:conv_net.best_epoch]
        conv_net.epoch_count   =  np.load(f"MIR_trained_models/{conv_net._name}/best_epoch.npy") +1 # cus we start at new epoch   


        conv_net.m_auc         =  conv_net.val_auc[conv_net.best_epoch-1]
        assert conv_net.best_epoch-1 > 10, "existing model was not trained for at least 10 epoch"
   
        print("best index of auc so far: ",np.argmax(conv_net.val_auc), " actual best epoch: ", conv_net.best_epoch-1) # cus epoch starts at count 1
    except:
            print("MIR_train.py: model did not use new param saving")
    return conv_net

if __name__ == "__main__":
    # model_name = "no_post_less_regular_noWI_73 auc 0.7644" # 100 epoch
    model_name = "no_postprocessing_regularisation" # 100 epoch
    
    
    # conv_net = new_model(model_name)
    conv_net = existing_model(model_name)
    conv_net.train_on_generator(model="no_postprocessing",batch_size=BATCH_SIZE,epochs=EPOCHS)
    

