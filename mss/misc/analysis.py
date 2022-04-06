import numpy as np
import matplotlib.pyplot as plt
# from a import VariationalAutoEncoder
from auto_encoder import VariationalAutoEncoder
from atrain import load_mnist
print(5/0)

def select_images(images,labels,num_images=10):
    sample_images_index = np.random.choice(range(len(images)),num_images)
    sample_images = images[sample_images_index]
    sample_labels = labels[sample_images_index]
    return sample_images,sample_labels

def plot_reconsrtucted_images(images,reconstructed_images):
    fig = plt.figure(figsize=(15,3))
    num_images = len(images)
   
    for i, (image,reconstructed_image) in enumerate(zip(images,reconstructed_images)):
        image = image.squeeze()
        ax = fig.add_subplot(2,num_images,i+1)
        ax.axis("off")
        ax.imshow(image,cmap="gray_r")
        reconstructed_image = reconstructed_image.squeeze()
        ax = fig.add_subplot(2,num_images,i+num_images + 1)
        ax.axis("off") 
        ax.imshow(reconstructed_image,cmap="gray_r")
    plt.show()

def plot_images_enconded_in_latent_space(latent_representations,sample_labels):
    plt.figure(figsize=(10,10))
    plt.scatter(latent_representations[:,0],
                latent_representations[:,1],
                cmap="rainbow",
                c=sample_labels,
                alpha=0.5,
                s=2)
    plt.colorbar()
    plt.show()

if __name__=="__main__":
    variational_auto_encoder = VariationalAutoEncoder.load("model_skipcon") #model_vae first_source_sep.wav
    variational_auto_encoder.summary()
    x_train,y_train,x_test,y_test = load_mnist()

    num_sample_images_to_show = 8
    sample_images,_ = select_images(x_test,y_test,num_sample_images_to_show) 
    reconstructed_images, _ = variational_auto_encoder.reconstruct(sample_images)
    plot_reconsrtucted_images(sample_images,reconstructed_images)
    
    # '''only works with latent_space = 2'''
    # num_images = 10000
    # sample_images, sample_labels = select_images(x_test,y_test,num_images) 
    # _, latent_representations = variational_auto_encoder.reconstruct(sample_images)
    # plot_images_enconded_in_latent_space(latent_representations,sample_labels)



