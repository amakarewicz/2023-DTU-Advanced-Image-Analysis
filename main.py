#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchgeometry.losses import dice
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy

# custom modules
import data
import model
#%% set up
root_dir = './EM_ISBI_Challenge/'
batch_size = 16
lr = 0.0001
epochs = 5
num_channels = 16

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
#%%
net = model.UNet(num_channels)
loss_function = torch.nn.BCELoss()
accuracy = BinaryAccuracy()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
#%%
train_loss_all = []
valid_loss_all = []
train_acc_all = []
valid_acc_all = []

n_train = len(train_loader)
n_valid = len(valid_loader)
for epoch in range(epochs):
    train_loss = 0
    train_acc = 0
    net.train()
    for x_batch, y_batch in train_loader:
        optimizer.zero_grad()
        output = net.forward(x_batch)
        y_pred = output[:,1:2,:,:]
        loss = loss_function(y_pred, y_batch)
        train_loss += loss.item()
        loss.backward()
        optimizer.step()
        train_acc += accuracy(y_pred, y_batch)
    train_loss_all.append(train_loss/n_train)
    train_acc_all.append(train_acc/n_train)
    net.eval()

    valid_loss = 0
    valid_acc = 0
    with torch.no_grad():
        for x_batch, y_batch in valid_loader:
            output = net.forward(x_batch)
            y_pred = output[:,1:2,:,:]
            val_loss = loss_function(y_pred, y_batch)
            valid_loss += val_loss.item()
            valid_acc += accuracy(y_pred, y_batch)
    valid_loss_all.append(valid_loss/n_valid)
    valid_acc_all.append(valid_acc/n_valid)

    # TODO: EarlyStopping

    print(f'Epoch {epoch} done, train_loss={train_loss_all[epoch]}, valid_loss={valid_loss_all[epoch]}, \
          train_acc={train_acc_all[epoch]}, valid_acc={valid_acc_all[epoch]}')
# %%
