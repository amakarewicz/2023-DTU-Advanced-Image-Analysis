#%%
import numpy as np
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
from torchgeometry.losses import dice
from torch.utils.data import DataLoader
from torchmetrics.classification import BinaryAccuracy, BinaryF1Score
import pandas as pd
# custom modules
import data
import model
import train
#%% set up
root_dir = './EM_ISBI_Challenge/'
batch_size = 32
lr = 3e-4
epochs = 150
num_channels = 32

if torch.cuda.is_available():
    device = torch.device("cuda:0")
    print('Using cuda')
else:
    device = torch.device('cpu')
    print('Using cpu')
#%%
# TODO: data augmentation (very basic so far)
# augmentation 2
img_trans = transforms.Compose([transforms.ToPILImage(),
                                  transforms.RandomCrop((128,128), padding=30, padding_mode='reflect'),
                                  transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
                                  transforms.RandomEqualize(p=0.3),
                                  transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                  transforms.ToTensor()
                                  ])

lab_trans = transforms.Compose([transforms.ToPILImage(),
                                  transforms.RandomCrop((128,128), padding=30, padding_mode='reflect'),
                                  transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
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

data_test = data.DataHandler(
    root_dir=root_dir,
    subset='test',
    return_patches=True) # no augmentation for the test set!
    # img_trans=img_trans,
    # lab_trans=lab_trans)
#%%
torch.manual_seed(123)
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
# ! different set-up
test_loader = DataLoader(dataset = data_test,
                         batch_size = int(batch_size/4),
                         shuffle = False,
                         num_workers = 0)
#%%
net = model.UNet(num_channels)
loss_function = torch.nn.BCELoss()
optimizer = torch.optim.Adam(net.parameters(), lr=lr, weight_decay=0.0001)
#%%
net, output = train.train_validation_loop(net, train_loader, valid_loader, epochs, optimizer, loss_function)
output_pd = pd.DataFrame(output)
output_pd.to_csv(f'results/results_final_train.csv', index=False)
#%%
import seaborn as sns
import cv2   
sns.set(style='white', font_scale=1.4)
for j, batch in enumerate(test_loader):
    x_batch, _ = batch
    y_pred = net.forward(x_batch)
    m = 3 #x_batch.shape[0]

    fig, ax = plt.subplots(2,2,figsize=(8,3.5*(2)))
    for i in range(0,m):
        ax[i][0].imshow(x_batch[i+1,0,:,:].detach().numpy(), cmap='gray')
        ax[0][0].set_title("Patch")
        # ax[i][1].imshow(np.round(t[i,0,:,:].detach().numpy()))
        # ax[i][1].set_title("True Segmentation")
        ax[i][1].imshow(np.round(y_pred[i+1,1,:,:].detach().numpy()), cmap='gray')
        ax[0][1].set_title("UNet Segmentation")
        cv2.imwrite(f'./results/test_img/img_{j+1}_batch_{i+1}.png', x_batch[i,0,:,:].detach().numpy())
        cv2.imwrite(f'./results/test_pred/pred_{j+1}_batch_{i+1}.png', np.round(y_pred[i,1,:,:].detach().numpy()))
    plt.tight_layout()
    plt.show()
    break
# %%
