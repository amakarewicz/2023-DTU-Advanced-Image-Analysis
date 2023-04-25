# TODO: Check how different padding affects the outputs!

import torch
import torch.nn as nn

class UNet(nn.Module):

    def __init__(self, num_channels: int = 16, padding='same'):
        super().__init__()
        self.n_ch = num_channels
        self.pad = padding

        self.relu = nn.ReLU(inplace=True)
        self.softmax = nn.Softmax(dim=1)
        self.pool = nn.MaxPool2d(kernel_size=2)    

        self.down_doubleconv1 = self.DoubleConv2d(1, self.n_ch, 3)
        self.down_doubleconv2 = self.DoubleConv2d(self.n_ch, self.n_ch*2, 3)
        self.down_doubleconv3 = self.DoubleConv2d(self.n_ch*2, self.n_ch*2*2, 3)

        self.up_trans32 = torch.nn.ConvTranspose2d(self.n_ch*2*2, self.n_ch*2, 2, stride=2) 
        self.up_trans21 = torch.nn.ConvTranspose2d(self.n_ch*2, self.n_ch, 2, stride=2)
        
        self.up_doubleconv2 = self.DoubleConv2d(self.n_ch*2*2, self.n_ch*2, 3)
        self.up_doubleconv1 = self.DoubleConv2d(self.n_ch*2, self.n_ch, 3)

        self.out_conv = nn.Conv2d(self.n_ch, 2, 1)
     

    def forward(self, x):
        down_conv1 = self.down_doubleconv1(x)
        x = self.pool(down_conv1)
        down_conv2 = self.down_doubleconv2(x)
        x = self.pool(down_conv2)
        down_conv3 = self.down_doubleconv3(x)
        x = torch.cat([self.up_trans32(down_conv3), down_conv2], dim=1)
        x = self.up_doubleconv2(x)
        x = torch.cat([self.up_trans21(x), down_conv1], dim=1)
        x = self.up_doubleconv1(x)
        x = self.softmax(self.out_conv(x))

        return x
    

    def DoubleConv2d(self, in_channels, out_channels, kernel_size):
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size, padding=self.pad),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size, padding=self.pad),
            nn.ReLU(inplace=True)
        )