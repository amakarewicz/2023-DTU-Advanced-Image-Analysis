#%%
import torch
import pandas as pd
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
# custom modules
import data
import model
import train
import augmentation
#%% set up
root_dir = './EM_ISBI_Challenge/'
# BATCH_SIZE = [32,64]
batch_size = 32
LR = [3e-4]#, 1e-4] # [1e-3, 1e-5, 
epochs = 200
NUM_CHANNELS = [32] # [16,32]#

#%%
for i, (img_trans, lab_trans) in tqdm(enumerate(zip(augmentation.IMG_TRANS[2:], augmentation.LAB_TRANS[2:])), desc='trans', position=0):
    data_train = data.DataHandler(
        root_dir=root_dir,
        subset='train',
        return_patches=True,
        img_trans=img_trans,
        lab_trans=lab_trans)
    torch.manual_seed(123)
    train_data, valid_data = torch.utils.data.random_split(data_train, (0.8, 0.2))
    train_loader = DataLoader(dataset = train_data,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = 0)
    valid_loader = DataLoader(dataset = valid_data,
                            batch_size = batch_size,
                            shuffle = False,
                            num_workers = 0)
    for lr in tqdm(LR, desc="lr", position=1):
        for num_channels in tqdm(NUM_CHANNELS, desc="ch", position=2):
            print(lr, num_channels, i)
            net = model.UNet(num_channels)
            loss_function = torch.nn.BCELoss()
            optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)

            net, output = train.train_validation_loop(net, train_loader, valid_loader, epochs, optimizer, loss_function)
            output_pd = pd.DataFrame(output)
            output_pd.to_csv(f'results/results_{epochs}_epochs{i+3}_trans_{num_channels}_ch_{lr:.0e}_lr.csv', index=False)
# %%
