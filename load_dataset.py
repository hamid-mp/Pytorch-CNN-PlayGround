
from sqlalchemy import true
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
import glob
import os
from PIL import Image
import cv2
from utils import preprocess
import numpy as np
class Custom_MNIST(Dataset):
  def __init__(self, data_folder, transform=None):
    self.transform = transform
    self.root = data_folder
    self.data = sorted( os.listdir(self.root) ) # get list of images
    self.labels = [int(x.split('_')[0]) for x in self.data] # read corresponding labels from image name
  def __len__(self):
    return len(self.labels) # get length of dataset

  def __getitem__(self, idx):
    #img = Image.open(os.path.join(self.root, self.data[idx])).convert('L')
    img = cv2.imread(os.path.join(self.root, self.data[idx]), 0)
    #img = cv2.resize(img, (32,32))
    #(thresh, img) = cv2.threshold(img, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    #img = np.float32(img)
    #img /= 255.0
    #img = 1 - img
    #img = preprocess(img)
    #img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    lbl = self.labels[idx]

    if self.transform: # apply image transforms if it is given
        img = self.transform(img)
    #im_bw = Image.fromarray(img)
    return ( img, torch.tensor(lbl) )

  def data_dist(self):
    '''
    get distribution of labels
    '''
    
    my_dict = {i:self.labels.count(i) for i in self.labels}
    return my_dict






'''

# calculate Mean and Std of dataset


mean = 0.
meansq = 0.
from tqdm import tqdm
for data, label in tqdm(trainset):
  mean = data.mean()
  meansq = (data**2).mean()


std = torch.sqrt(meansq - mean**2)
print()
print("mean: " + str(mean))
print("std: " + str(std))
print()


mean = 0.
meansq = 0.
for data, label in tqdm(validset):
    mean = data.mean()
    meansq = (data**2).mean()

std = torch.sqrt(meansq - mean**2)
print()
print("mean: " + str(mean))
print("std: " + str(std))
print()
'''

