import torch
import torchvision
import matplotlib.pyplot as plt
import cv2
import numpy as np
import aif_model.utils as utils

images_root = 'datasets/'+"32closer"+'/'
orientation_path = 'datasets/'+'32closer'+'/centroids.csv'
ds = utils.ImageDataset(images_root, orientation_path)
train_gen, test_gen = utils.split_dataset(ds, percent=0.9)

def get_val_imgs(n):
    val_imgs = [None] * n
    lat_reps = [None] * n
    counter = 0
    for x, y in test_gen:
        for x_, y_ in zip(x, y):
            val_imgs[counter] = x_
            lat_reps[counter] = y_
            counter+=1
            if counter == n: return torch.stack(val_imgs), torch.stack(lat_reps)

vae = torch.load("vae.pt")

n = 5
# get representation
val_imgs, lat_reps = get_val_imgs(n)
mu, logvar = vae.encoder(val_imgs)

# reconstruct
eps = vae.prior().sample(mu.shape)
z = mu + logvar.exp() * eps
mu_x, _ = vae.decoder(z)

for i in range(n):
    print("Original")
    plt.imshow(val_imgs[i].detach().permute((1,2,0)))
    plt.show()
    print(mu[i])

    print("Reconstruction")
    plt.imshow(mu_x[i].detach().permute((1,2,0)))
    plt.show()

mu = mu.detach()
logvar = logvar.detach()
print("Moving second element from -1 to 1")
for i in [-1, -0.5, 0, 0.5, 1]:
    mu[0][0] = i
    eps = vae.prior().sample(mu.shape)
    eps = eps.detach()
    z = mu + logvar.exp() * eps
    mu_x, _ = vae.decoder(z)

    print(mu[0])
    plt.imshow(mu_x[0].detach().permute((1,2,0)))
    plt.show()
