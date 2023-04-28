import torchvision.transforms as transforms
import torch

# img_trans = transforms.Compose([transforms.ToPILImage(),
#                                 transforms.RandomRotation(degrees = 45),
#                                 transforms.ToTensor(),
#                                 transforms.Normalize((0),(1))
#                                 ])

# lab_trans = transforms.Compose([transforms.ToPILImage(),
#                                 transforms.RandomRotation(degrees = 45),
#                                 transforms.ToTensor()
#                                 ])

# data augmentation
class AddGaussianNoise(object):
    '''
    Source: https://discuss.pytorch.org/t/how-to-add-noise-to-mnist-dataset-when-using-pytorch/59745
    '''
    def __init__(self, mean=0., std=1.):
        self.std = std
        self.mean = mean
        
    def __call__(self, tensor):
        return tensor + torch.randn(tensor.size()) * self.std + self.mean
    
    def __repr__(self):
        return self.__class__.__name__ + '(mean={0}, std={1})'.format(self.mean, self.std)

# no augmentation
img_trans_1 = transforms.ToTensor()
lab_trans_1 = transforms.ToTensor()

# augmentation 1
img_trans_2 = transforms.Compose([transforms.ToPILImage(),
                                  transforms.RandomRotation(degrees = 30),
                                  transforms.CenterCrop((96,96)),
                                  transforms.RandomHorizontalFlip(p=0.3),
                                  transforms.RandomVerticalFlip(p=0.3),
                                  transforms.ToTensor(),
                                  transforms.Normalize((0),(1)),
                                  AddGaussianNoise(std=0.05)
                                  ])

lab_trans_2 = transforms.Compose([transforms.ToPILImage(),
                                  transforms.RandomRotation(degrees = 30),
                                  transforms.CenterCrop((96,96)),
                                  transforms.RandomHorizontalFlip(p=0.3),
                                  transforms.RandomVerticalFlip(p=0.2),
                                  transforms.ToTensor()
                                  ])

# augmentation 2
img_trans_3 = transforms.Compose([transforms.ToPILImage(),
                                  transforms.RandomCrop((128,128), padding=30, padding_mode='reflect'),
                                  transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
                                  transforms.RandomEqualize(p=0.3),
                                  transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5)),
                                  transforms.ToTensor()
                                  ])

lab_trans_3 = transforms.Compose([transforms.ToPILImage(),
                                  transforms.RandomCrop((128,128), padding=30, padding_mode='reflect'),
                                  transforms.RandomPerspective(distortion_scale=0.6, p=0.5),
                                  transforms.ToTensor()
                                  ])

IMG_TRANS = [img_trans_1, img_trans_2, img_trans_3]
LAB_TRANS = [lab_trans_1, lab_trans_2, lab_trans_3]