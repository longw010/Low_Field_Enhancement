####
import glob
import os
import numpy as np
from tqdm import tqdm
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from PIL import Image

myTransforms = transforms.Compose([
    transforms.ToTensor()
])


class MyDataSet(Dataset):
    def __init__(self, mode='train'):

        self.raw_data = load_npz_pro(mode=mode)
        self.images1 = self.raw_data['img']
        self.labels = self.raw_data['lab']
        # self.transf = myTransforms
        self.lenth = self.images1.shape[0]

    def __getitem__(self, index):

        img1 = self.images1[index]
        lab = self.labels[index]

        img1 = myTransforms(img1)
        lab = myTransforms(lab)

        return img1, lab

    def __len__(self):
        return self.lenth


def getdataset(batchsize=1):
    DS_train = MyDataSet(mode='train')
    DS_test = MyDataSet(mode='test')

    DS_train_loader = DataLoader(
        dataset=DS_train, batch_size=batchsize, shuffle=True)
    DS_test_loader = DataLoader(
        dataset=DS_test, batch_size=batchsize, shuffle=True)

    return DS_train_loader, DS_test_loader
