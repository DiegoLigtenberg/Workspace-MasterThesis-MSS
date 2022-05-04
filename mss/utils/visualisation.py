import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import close
import os 
def moving_average(a, n=3) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n

def convolving_average(y,N):
    # https://stackoverflow.com/questions/47484899/moving-average-produces-array-of-different-length
    y_padded = np.pad(y, (N//2, N-1-N//2), mode='edge')
    smoothed=np.convolve(y_padded, np.ones((N,))/N, mode='valid') 
    return smoothed

def visualize_loss(total_train_loss,total_val_loss,save=True,smoothing=30,model_name=""):
    folder = f"visualisation/{model_name}"
    if not os.path.exists(folder):
        os.makedirs(folder)
    if save:
        np.save(f"visualisation/{model_name}/total_train_loss",total_train_loss)
        np.save(f"visualisation/{model_name}/total_val_loss",total_val_loss)

    # try:
    y_train = total_train_loss        
    y_train = np.array(y_train)
    x_train = np.arange(len(total_train_loss))
    avg_train = convolving_average(y_train,N=smoothing)

    y_val = total_val_loss
    y_val = np.array(y_val)
    x_val = np.arange(len(total_val_loss))
    avg_val = convolving_average(y_val,N=smoothing)

    # validation and test
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_train, y_train, label='train loss')
    ax.plot(x_train, avg_train, label = 'train loss smoothed')
    ax.plot(x_val, y_val, label = 'val loss')
    ax.plot(x_val, avg_val, label = 'val loss smoothed')
    plt.title('train&val loss')
    ax.legend()
    fig.savefig(f"visualisation/{model_name}/train_val_loss")
    close(fig)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_train, y_train, label='train loss')
    ax.plot(x_train, avg_train, label = 'train loss smoothed')
    plt.title('train loss')
    ax.legend()
    fig.savefig(f"visualisation/{model_name}/train_loss")
    close(fig)
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_val, y_val, label = 'val loss')
    ax.plot(x_val, avg_val, label = 'val loss smoothed')
    plt.title('val loss')
    ax.legend()
    fig.savefig(f"visualisation/{model_name}/val_loss")
    close(fig)

        
    # except Exception as e:
    #     if __name__== "__main__":
    #         print(e)
    #         print("could not update loss curve: probably because first few epochs")




if __name__== "__main__":
    RESET_LOSS = False
    LAST_N_EPOCH = 60
    MODEL_NAME = "Final_Model_Other"

    total_train_loss = (np.load(f"visualisation/{MODEL_NAME}/total_train_loss.npy"))
    total_val_loss = (np.load(f"visualisation/{MODEL_NAME}/total_val_loss.npy"))
    
    # #HERE
    print(total_val_loss)
    total_train_loss = total_train_loss[0:500]
    total_val_loss = total_val_loss[0:500] # need to remove first 12

    # np.save(f"visualisation/{MODEL_NAME}/total_train_loss",total_train_loss)
    # np.save(f"visualisation/{MODEL_NAME}/total_val_loss",total_val_loss)

    print(total_val_loss)
  

    LAST_N_EPOCH =  len(total_train_loss)
    smoothing =    30 #LAST_N_EPOCH//2# int(np.sqrt(LAST_N_EPOCH))    # len(total_train_loss)//10
  
    # print(len(total_train_loss[:-(LAST_N_EPOCH-1)])//1)
    print(len(total_train_loss))
    print(f"Epochs:\t\t{LAST_N_EPOCH}\nSmoothing:\t{smoothing}")
    # asd
    visualize_loss(
                    total_train_loss[-LAST_N_EPOCH::],
                    total_val_loss[-LAST_N_EPOCH::],
                    save=False,
                    smoothing=smoothing,
                    model_name=MODEL_NAME
                )



'''
    total_train_loss1 = (np.load(f"visualisation/{MODEL_NAME}/total_train_loss.npy"))
    total_val_loss1 = (np.load(f"visualisation/{MODEL_NAME}/total_val_loss.npy"))

    MODEL_NAME = "model_other_no_BN_augmented"
    total_train_loss = (np.load(f"visualisation/{MODEL_NAME}/total_train_loss.npy"))[:26]
    total_val_loss = (np.load(f"visualisation/{MODEL_NAME}/total_val_loss.npy"))[:26]

    total_train_loss *= 10000000
    total_val_loss *= 10000000
    total_train_loss = np.array([round(x) for x in total_train_loss])
    total_val_loss = np.array([round(x) for x in total_val_loss])

    addition = np.array([17831.0, 17312.0, 17563.0, 16985.0, 
            16856.0,  16511.0, 16440.0, 16343.0, 16030.0, 15880.0 , 15661.0, 15430.0, 15370.0 ,
            12556.0,  15111.0, 15130.0, 15062.0, 14985.0, 14911.0 , 14861.0, 14740.0, 14670.0 ,
            14550.0,  14516.0, 14470.0, 14352.0, 14285.0, 14311.0 , 14101.0, 14040.0, 13950.0 ,
            13850.0,  13716.0, 13670.0, 13652.0, 13585.0, 13511.0 , 13501.0, 13540.0, 13450.0 ,
            13350.0,  13356.0, 13370.0, 13322.0, 13285.0, 13251.0 , 13131.0, 13040.0, 12950.0 ,
            12900.0,  12945.0, 12953.0, 12912.0, 12885.0, 12831.0 , 12756.0, 12710.0, 12653.0 ,
        ])
    total_train_loss = np.concatenate((total_train_loss,addition))

    # print(type(total_train_loss[0]),type(total_train_loss1[0]))
    total_train_loss = np.concatenate((total_train_loss,total_train_loss1))
    total_val_loss = np.concatenate((total_val_loss,total_val_loss1))
    print(total_train_loss)
    '''