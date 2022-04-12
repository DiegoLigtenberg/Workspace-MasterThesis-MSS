import matplotlib.pyplot as plt
import numpy as np

def visualize_loss(total_train_loss,total_val_loss):
    y = total_train_loss
    x = np.arange(len(total_train_loss))
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, y, label='$y = numbers')
    plt.title('train loss')
    ax.legend()
    fig.savefig("visualisation/train_loss")

    y = total_val_loss
    x = np.arange(len(total_val_loss))
    fig = plt.figure()
    ax = plt.subplot(111)
    ax.plot(x, y, label='$y = numbers')
    plt.title('val loss')
    ax.legend()
    fig.savefig("visualisation/val_loss")
    
    plt.close()

    np.save("visualisation/total_train_loss",total_train_loss)
    np.save("visualisation/total_val_loss",total_val_loss)