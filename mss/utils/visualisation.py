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
    MODEL_NAME = "model_instruments_regularized"

    total_train_loss = list(np.load(f"visualisation/{MODEL_NAME}/total_train_loss.npy"))
    total_val_loss = list(np.load(f"visualisation/{MODEL_NAME}/total_val_loss.npy"))

    LAST_N_EPOCH = len(total_train_loss)-1
    smoothing =    int(np.sqrt(LAST_N_EPOCH))    # len(total_train_loss)//10
    # print(len(total_train_loss[:-(LAST_N_EPOCH-1)])//1)

    print(f"Epochs:\t\t{LAST_N_EPOCH}\nSmoothing:\t{smoothing}")
    visualize_loss(
                    total_train_loss[-LAST_N_EPOCH::],
                    total_val_loss[-LAST_N_EPOCH::],
                    save=False,
                    smoothing=smoothing,
                    model_name=MODEL_NAME
                )
