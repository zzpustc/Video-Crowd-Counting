import numpy as np
import os
import random
from scipy import io as sio
import torch
from torch.utils import data
from PIL import Image, ImageOps

import pandas as pd

from config import cfg

class FDST(data.Dataset):
    def __init__(self, data_path, mode, main_transform=None, img_transform=None, gt_transform=None):
        self.img_path = data_path + '/img'
        self.gt_path = data_path + '/den'
        self.data_files = os.listdir(self.img_path)
        self.data_files.sort(key=lambda x: int(x.split('_')[0])*1000+ int(x.split('_')[1].split('.')[0]))
        self.num_samples = len(self.data_files) 
        self.main_transform=main_transform  
        self.img_transform = img_transform
        self.gt_transform = gt_transform

        self.mode = mode

        if self.mode is 'train':
            print('[FDST DATASET]: %d training images.'  % (self.num_samples))
        if self.mode is 'test':
            print('[FDST DATASET]: %d testing images.'  % (self.num_samples))     
    
    def __getitem__(self, index):
        # print self.data_files[index]
        sign = index / 150
        num = index - sign * 150
        if num == 0:
            index += 1
        pname = self.data_files[index]
        fname = self.data_files[index-1]
        pimg, pden = self.read_image_and_gt(pname) 
        fimg, fden = self.read_image_and_gt(fname) 
        if self.main_transform is not None:
            pimg, pden = self.main_transform(pimg,pden) 
            fimg, fden = self.main_transform(fimg,fden)
        if self.img_transform is not None:
            pimg = self.img_transform(pimg)  
            fimg = self.img_transform(fimg) 
        if self.gt_transform is not None:
            pden = self.gt_transform(pden)  
            fden = self.gt_transform(fden) 
        return pimg, pden, fimg, fden

    def __len__(self):
        return self.num_samples

    def read_image_and_gt(self,fname):
        img = Image.open(os.path.join(self.img_path,fname))
        if img.mode == 'L':
            img = img.convert('RGB')

        # den = sio.loadmat(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.mat'))
        # den = den['map']
        den = pd.read_csv(os.path.join(self.gt_path,os.path.splitext(fname)[0] + '.csv'), sep=',',header=None).values
        
        den = den.astype(np.float32, copy=False)    
        den = Image.fromarray(den)  
        return img, den    

    def get_num_samples(self):
        return self.num_samples       
            
        
