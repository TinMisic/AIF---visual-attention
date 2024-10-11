import aif_model.utils as utils
import torch
from torch.utils import data
import aif_model.config as c
import numpy as np
import cv2

def find_ball(img):
    red = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    red[(img[:,:,2] > 200)] = 255
    green = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    green[(img[:,:,1] < 200)] = 255
    blue = np.zeros((img.shape[0], img.shape[1]), dtype=np.uint8)
    blue[(img[:,:,0] < 200)] = 255
    
    binary = np.logical_and(np.logical_and(red == 255, green == 255), blue == 255).astype(np.uint8) * 255
    
    # Find contours in the binary image
    contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    cX,cY,area = -1,-1,-1
    # Iterate through contours
    for contour in contours:
        # Calculate moments of the contour
        M = cv2.moments(contour)
        
        # Calculate centroid
        if M["m00"] != 0:
            cX = 2*(M["m10"] / M["m00"])/img.shape[1] - 1
            cY = 2*(M["m01"] / M["m00"])/img.shape[0] - 1
            area = M["m00"] / (img.shape[0]*img.shape[1])
        break
    return (cX,cY,area)

images_root = 'datasets/'+"32closer"+'/'
orientation_path = 'datasets/'+'32closer'+'/centroids.csv'
ds = utils.ImageDataset(images_root, orientation_path)
gen = data.DataLoader(ds,num_workers=4)

vae = torch.load("vae.pt")
data = []
i = 0
for x, y in gen:
    i+=1
    print(i/200,"%")
    centroid = find_ball(np.transpose(x.detach().numpy().squeeze(),(1,2,0)))
    tMin = torch.min(x)
    tMax = torch.max(x) - tMin
    tMax = max(tMax, 1e-8)
    x = (x - tMin) / (tMax) # Scaling to [0, 1]
    mu, log_var = vae.encoder(x)
    mu = np.squeeze(mu.detach().numpy())
    data.append(np.concatenate((mu,centroid[:2])))

data = np.array(data)
print(data.shape)
np.savetxt("translation.csv", data, delimiter=',')
