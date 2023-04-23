#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchgeometry.losses import dice
from torch.utils.data import DataLoader
# custom modules
import data
# TODO: import model
#%% set up
root_dir = './EM_ISBI_Challenge/'
batch_size = 16
# lr = 0.0001
# epochs = 10

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Using cuda')
else:
    device = torch.device('cpu')
    print('Using cpu')
#%%
# TODO: data augmentation (very basic so far)
img_trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomRotation(degrees = 45),
                                transforms.ToTensor(),
                                transforms.Normalize((0),(1))
                                ])

lab_trans = transforms.Compose([transforms.ToPILImage(),
                                transforms.RandomRotation(degrees = 45),
                                transforms.ToTensor()
                                ])
#%%
# training data
data_train = data.DataHandler(
    root_dir=root_dir,
    subset='train',
    return_patches=True,
    img_trans=img_trans,
    lab_trans=lab_trans)

#%%
train_data, valid_data = torch.utils.data.random_split(data_train, (0.8, 0.2))
#%%
# dataloaders
train_loader = DataLoader(dataset = train_data,
                          batch_size = batch_size,
                          shuffle = False,
                          num_workers = 0)
valid_loader = DataLoader(dataset = valid_data,
                          batch_size = batch_size,
                          shuffle = False,
                          num_workers = 0)