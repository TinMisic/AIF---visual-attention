import torch
from torch.utils import data
import numpy as np
from pathlib import Path
from PIL import Image
import torchvision
import aif_model.config as c
import cv2 as cv

# Define image dataset class
class ImageDataset(data.Dataset):
    def __init__(self, images_root, orientation_path, size = 200000):
        self.imgPaths = list(Path(images_root).rglob('img*.jpg'))
        self.orientation = np.genfromtxt(orientation_path, delimiter=',')
        length = min(size,len(self.imgPaths))

        self.orientation = self.orientation[:length,:]
        self.imgPaths = self.imgPaths[:length]

    def __len__(self):
        return len(self.imgPaths)

    def __getitem__(self, i):
        sample = None
        with Image.open(self.imgPaths[i]) as img:
            try:
                #img = img.resize((c.width,c.height))
                t = torchvision.transforms.functional.pil_to_tensor(img)
                tMin = 0#torch.min(t)
                tMax = 255#torch.max(t) - tMin
                tMax = max(tMax, 1e-8)
                t = (t - tMin) / (tMax) # Scaling to [0, 1]
                sample=t
            except OSError:
                return self.__getitem__(i-1) # return previous image
            
        lat_rep = np.zeros((c.latent_size))
        # force latent representation
        # lat_rep[0] = self.orientation[i][0]
        # lat_rep[1] = self.orientation[i][1]
        # lat_rep[2] = self.orientation[i][2]

        return sample, lat_rep
    
# Define intention dataset class
class IntentionDataset(data.Dataset):
    def __init__(self, data_path):
        datas = np.loadtxt(data_path, delimiter=',')
        X = datas[:, :5]  # features
        y = datas[:, 5:]  # labels

        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
    
# Split dataset in train/test set and in batches
def split_dataset(dataset, percent):
    length = int(dataset.__len__()*percent)
    train_set, test_set = data.random_split(dataset, (length, dataset.__len__() - length))
    train_gen = data.DataLoader(train_set, batch_size=c.n_batch, shuffle=True, num_workers=4)
    test_gen = data.DataLoader(test_set, batch_size=c.n_batch,num_workers=4)

    return train_gen, test_gen

# Kullback–Leibler divergence
def kl_divergence(p_m, p_v, q_m, log_q_v):
    return torch.mean(0.5 * torch.sum(torch.log(p_v) - log_q_v + (log_q_v.exp() + (q_m - p_m) ** 2) / p_v - 1, dim=1), dim=0)

# Shifts rows down by n rows
def shift_rows(matrix, n):
    shifted_matrix = np.concatenate((matrix[-n:], matrix[:-n]), axis=0)
    return shifted_matrix

def pixels_to_angles(coordinates):
    """
    Translates pixel coordinates into angles in radians
    """
    f = c.width / (2 * np.tan(c.horizontal_fov/2))
    
    cent = (c.width/2, c.height/2) # get center point

    # calculation
    u = -(coordinates[:,0] - cent[0]) # negative because of axis of rotation
    v = coordinates[:,1] - cent[1]

    yaw = np.rad2deg(np.arctan2(u, f))
    pitch = np.rad2deg(np.arctan2(v, f))

    return np.vstack((pitch, yaw)).T # first pitch then yaw

# Normalize angles
def normalize(x):
    return x / c.width * 2 - 1


# Denormalize angles
def denormalize(x):
    return (x + 1) / 2 * c.width

# Gaussian noise
def add_gaussian_noise(array):
    sigma = c.noise ** 0.5
    return array + np.random.normal(0, sigma, np.shape(array))

# display vectors on image
def display_vectors(img, vectors):

    h,w,c = img.shape

    red = (w//2 + int(vectors[0,0]*w/2),h//2+int(vectors[0,1]*h/2))
    blue = (w//2 + int(vectors[1,0]*w/2),h//2+int(vectors[1,1]*h/2))

    arrowed = cv.arrowedLine(img.copy(), (w//2,h//2),red,(150,0,0),2)
    arrowed = cv.arrowedLine(arrowed, (w//2,h//2),blue,(0,0,150),2)

    return arrowed

