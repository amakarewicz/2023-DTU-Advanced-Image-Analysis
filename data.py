import numpy as np
import matplotlib.pyplot as plt
import torch
import skimage.io
import os
import torchvision.transforms as transforms


class DataHandler(torch.utils.data.Dataset):
    
    def __init__(self,
                 root_dir = 'EM_ISBI_Challenge',
                 subset='train',
                 return_patches=True,
                 seed=123,
                 patch_size=128,
                 img_trans= transforms.Compose([transforms.ToTensor()]),
                 lab_trans= transforms.Compose([transforms.ToTensor()])):
        self.images = []
        self.labels = []
        self.img_trans = img_trans
        self.lab_trans = lab_trans
        self.return_patches = return_patches

        self.seed = seed
        np.random.seed(seed)
        # paths
        img_root_dir = os.path.join(root_dir, f'{subset}_images')
        lab_root_dir = os.path.join(root_dir, f'{subset}_labels')
        img_paths = os.listdir(img_root_dir)
        lab_paths = ['' for i in range(len(img_paths))]
        if subset=='train': 
            lab_paths = os.listdir(lab_root_dir)
        
        # loading full images
        for img, lab in zip(img_paths, lab_paths):
            img = skimage.io.imread(os.path.join(img_root_dir, img))
            img = 1.0 - torch.tensor(img, dtype=torch.float32)/255.0
            self.images.append(img)

            if lab != '':
                lab = skimage.io.imread(os.path.join(lab_root_dir, lab))
                lab = 1.0 - torch.tensor(lab, dtype=torch.float32)/255.0
            else: 
                lab = torch.empty(img.shape, dtype=torch.float32)
            self.labels.append(lab)
        
        if self.return_patches:
            self.images_patches=[]
            self.labels_patches=[]
            self.patch_size = patch_size
            # smaller patches
            for img, lab in zip(self.images, self.labels):
                img_patch = self.get_patch(img)
                lab_patch = self.get_patch(lab)
                for j in range(len(img_patch)):
                    self.images_patches.append(np.array(img_patch[j],dtype=np.float32))
                    self.labels_patches.append(np.array(lab_patch[j],dtype=np.float32))

    def __getitem__(self, idx):
        img = self.images_patches[idx] if self.return_patches else self.images[idx]
        lab = self.labels_patches[idx] if self.return_patches else self.labels[idx]
        # setting the seed needed prior to any transform
        torch.manual_seed(self.seed)
        img = self.img_trans(img)
        torch.manual_seed(self.seed)
        lab = self.lab_trans(lab)
        return img, lab
    
    def __len__(self):
        length = len(self.images)
        if self.return_patches:
            length = len(self.images_patches)
        return length
    
    def get_patch(self, img):
        return (img
                .unfold(0, self.patch_size, self.patch_size)
                .unfold(1, self.patch_size, self.patch_size)
                .flatten(0,1))
