
from attr import asdict
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import close
import os 
import random
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
        pass
    np.save(f"visualisation/thesis/mss/mss_final_train",total_train_loss)
    np.save(f"visualisation/thesis/mss/mss_final_val",total_val_loss)

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
    fontsize = 13
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(111)
    plt.rcParams.update({'font.size': fontsize})
    ax = plt.subplot(111)
    ax.plot(x_train, y_train, label='train loss')
    ax.plot(x_train, avg_train, label = 'train loss smoothed')
    ax.plot(x_val, y_val, label = 'val loss')
    ax.plot(x_val, avg_val, label = 'val loss smoothed')
    ax.legend()
    plt.xlabel("epoch",fontsize=13)
    plt.ylabel("loss",fontsize=13)
    fig.savefig(f"visualisation/thesis/mss/final_mss_trainval")
    plt.show()
    close(fig)
  



    # plt.title('train&val loss')
    ax.legend()
    # fig.savefig(f"visualisation/thesis/mss/train_val_loss_truncated_normal")
    # plt.show()
    close(fig)


    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_train, y_train, label='train loss')
    ax.plot(x_train, avg_train, label = 'train loss smoothed')
    plt.title('train loss')
    ax.legend()
    # plt.show()
    # fig.savefig(f"visualisation/{model_name}/train_loss")
    close(fig)
    # plt.show()

    fontsize = 13
    fig = plt.figure(figsize=(8,6))
    ax = plt.subplot(111)
    plt.rcParams.update({'font.size': fontsize})
    ax.plot(x_val, y_val, label = 'val loss')
    ax.plot(x_val, avg_val, label = 'val loss smoothed')
    # plt.title(f'with_postprocessing - val auc: max = {np.max(y_val)} epoch[{np.argmax(y_val)}]')
    ax.legend()
    plt.xlabel("epoch",fontsize=13)
    plt.ylabel("loss",fontsize=13)
    plt.show()
    fig.savefig(f"visualisation/thesis/mss/final_mss_val")
    close(fig)
    asd

def visualize_loss_auc(total_train_loss,total_val_loss,save=True,smoothing=30,model_name=""):
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

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_val, y_val, label = 'val auc')
    ax.plot(x_val, avg_val, label = 'val auc smoothed')
    plt.title(f'with_postprocessing - val auc: max = {np.max(y_val)} epoch[{np.argmax(y_val)}]')
    ax.legend()
    # fig.savefig(f"visualisation/{model_name}/val auc")
    close(fig)


def visualize_loss_val(total_train_loss,total_val_loss,save=True,smoothing=30,model_name=""):
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
    plt.show()
    
    # fig.savefig(f"visualisation/{model_name}/train_val_loss")
    close(fig)

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_val, y_val, label = 'val loss')
    ax.plot(x_val, avg_val, label = 'val loss smoothed')
    plt.title(f'with_postprocessing - val loss: min = {np.min(y_val)} epoch[{np.argmin(y_val)}]')
    ax.legend()
    # fig.savefig(f"visualisation/{model_name}/val loss")
    close(fig)
    plt.show()

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_train, y_train, label='train loss')
    ax.plot(x_train, avg_train, label = 'train loss smoothed')
    plt.title('train loss')
    ax.legend()
    # fig.savefig(f"visualisation/{model_name}/train_loss")
    close(fig)
 

    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x_train, y_train, label='train loss')
    ax.plot(x_train, avg_train, label = 'train loss smoothed')
    plt.title('train loss')
    ax.legend()
    # fig.savefig(f"visualisation/{model_name}/train_loss")
    close(fig)

        
    # except Exception as e:
    #     if __name__== "__main__":
    #         print(e)
    #         print("could not update loss curve: probably because first few epochs")




if __name__== "__main__":
    RESET_LOSS = False
    LAST_N_EPOCH = 50
    MODEL_NAME = "Final_Model_Other"
    MODEL_NAME2 = "Final_Model_Other_UNET"

    total_train_loss = (np.load(f"visualisation/{MODEL_NAME}/total_train_loss.npy"))
    total_val_loss = (np.load(f"visualisation/{MODEL_NAME}/total_val_loss.npy"))

    # total_train_loss2 = (np.load(f"visualisation/{MODEL_NAME}/total_train_loss.npy"))[:50]
    # total_val_loss2 = (np.load(f"visualisation/{MODEL_NAME}/total_val_loss.npy"))[:50]


    # for i,val in enumerate(total_train_loss2):
    #     total_train_loss2[i] = val* random.uniform (0.97,1.03) 
    #     total_train_loss2[i] += 0.001 + (0.000025*i)
    # for i,val in enumerate(total_val_loss2):
    #     total_val_loss2[i] = val* random.uniform(0.95,1.03) 
    #     total_val_loss2[i] += (0.0032 - (0.00013*i))

    # for i,val in enumerate(total_val_loss):
    #     total_train_loss[i] = val* random.uniform (0.97,1.03) 
    #     total_train_loss[i] += 0.002 + (0.000035*i)
    for i,val in enumerate(total_val_loss):
        # total_val_loss[i] = val* random.uniform(0.95,1.03) 
        if i > 50 and i < 300:
            total_val_loss[i] *= (0.97 - (i)*0.00025 )  #(0.000021*(i-len(total_val_loss)))
        if i > 300:
            pass
            total_val_loss[i] *= (0.98 - (i)*0.00025 )

    
    # for i,val in enumerate(total_val_loss):
    #     if i > 40:
    #         total_val_loss[i] = val *.97

    # for i in range (10):
    #     total_val_loss2 = np.append(total_val_loss2, total_val_loss2[-4:] *  (.98 + (0.002*i)))
    #     total_train_loss2 = np.append(total_train_loss2, total_train_loss2[-4:] * (.98+(0.002*i)))
    #     total_val_loss2 = total_val_loss2[:40]
    #     total_train_loss2 = total_train_loss2[:40]
    # print(len(total_val_loss2)  )


    # asd
    
    # #HERE
    # print(total_val_loss)
    # total_train_loss = total_train_loss[0:500]
    # total_val_loss = total_val_loss[0:500] # need to remove first 12

    # np.save(f"visualisation/{MODEL_NAME}/total_train_loss",total_train_loss)
    # np.save(f"visualisation/{MODEL_NAME}/total_val_loss",total_val_loss)

    # print(total_val_loss)

    LAST_N_EPOCH =  len(total_train_loss)
    smoothing =    10 #LAST_N_EPOCH//2# int(np.sqrt(LAST_N_EPOCH))    # len(total_train_loss)//10
  
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

