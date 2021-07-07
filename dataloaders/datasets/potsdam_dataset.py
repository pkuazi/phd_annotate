from __future__ import print_function, division
import numpy as np
import pandas as pd
import time, os, sys, json, math, re, random
import threading, logging, copy, scipy
from datetime import datetime, timedelta
# from config import config
import torch
import torch.nn as nn
from torch.utils.data.dataset import Dataset
import torch.nn.functional as F
import torchvision.transforms as transforms
from torch.autograd import Variable
#import torch.nn.functional as functional
import cv2
import itertools
import gdal
from PIL import Image
from mypath import Path
from dataloaders import custom_transforms as tr

random.seed(20201020)

# img_path = '/data/data/AIDataset/img'
# gt_path = '/data/data/AIDataset/gt'
img_path = '/home/zjh/AIDataset/img_jpg/'
gt_path = '/home/zjh/AIDataset/gt_tif'

BLOCK_SIZE = 256
OVERLAP_SIZE = 0

classid2rgb_map = {
    0: [255, 255, 255],  # "white", 地面
    1: [255, 0, 0],  # "red",房屋
    2: [0, 0, 255],  # "blue", 其他
    3: [0, 255, 255],  # "cyan", 车
    4: [0, 255, 0],  # "green", 树木
    5: [255, 255, 0],  # "yello" 草地
}

class Imgdataset(Dataset):
    
    NUM_CLASSES = 6
    
    def __init__(self, args, files_names,files_root,gt_root,BLOCK_SIZE, shuffle = False, split = 'train'):
        super().__init__()
        logging.info("ImgloaderPostdam->__init__->begin:")
        random.seed(20201023)

        self.img_path = files_root
        self.gt_path = gt_root
        self.shuffle = shuffle
        self.file_names = files_names
        self.split = split
        self.args = args
        self.BLOCK_SIZE=BLOCK_SIZE
    
    def __len__(self):
        return len(self.file_names)

    def __getitem__(self, idx):
        if self.shuffle:
            idx = random.sample(range(len(self.file_names)), 1)[0]
        
        filename,xoff, yoff = self.file_names[idx]
        
        r=filename.split('_')[2]
        c=filename.split('_')[3]
        gtname='top_potsdam_%s_%s_label.tif'%(r,c)
        # gtname='osm_buildings_%s_%s.tif'%(r,c)

        imgfile = os.path.join(self.img_path, filename)
        gtfile = os.path.join(self.gt_path, gtname)
        
        imgds = gdal.Open(imgfile,gdal.GA_ReadOnly)
        gtds = gdal.Open(gtfile,gdal.GA_ReadOnly)
        if gtds is None:
            print('Failed to open file:',gtds)
            sys.exit(1)
        img_tif = imgds.ReadAsArray(xoff, yoff, self.BLOCK_SIZE, self.BLOCK_SIZE).astype(np.float32)
        mask = gtds.ReadAsArray(xoff, yoff, self.BLOCK_SIZE, self.BLOCK_SIZE)
        
        img_tif[np.isnan(img_tif)] = 0
        imax = img_tif.max()
        imin = img_tif.min()
        img_tif = (img_tif - imin)/(imax - imin)
        img_tif *= 255
        tmp = np.zeros(list(mask.shape))
        tmp[mask==1]=1
        tmp[mask==2]=2
        tmp[mask==3]=3
        tmp[mask==4]=4
        tmp[mask==5]=5
        mask=tmp
        # mask[np.isnan(mask)] = 255
        # label 1 is building
        # mask[mask == 2] = 0
        # mask[mask == 3] = 0
        # mask[mask == 4] = 0
        # mask[mask == 5] = 0
        
        img = torch.from_numpy(img_tif).float()
        mask = torch.from_numpy(mask).float()
        
        # sample = {'image': img, 'label': gtds}

        return {'image': img, 'label': mask}
        # return img, mask, idx# for truncated loss

        # if self.split == "train":
        #     return self.transform_tr(sample)
        # elif self.split == 'val':
        #     return self.transform_val(sample)
        # elif self.split == 'test':
        #     return self.transform_ts(sample)
        
    # def transform_tr(self, sample):
    #     composed_transforms = transforms.Compose([
    #         tr.RandomHorizontalFlip(),
    #         tr.RandomScaleCrop(base_size=self.args.base_size, crop_size=self.args.crop_size),    # , fill=255
    #         tr.RandomGaussianBlur(),
    #         # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         tr.ToTensor()])
    #     return composed_transforms(sample)
    
    # def transform_val(self, sample):
    #     composed_transforms = transforms.Compose([
    #         tr.FixScaleCrop(crop_size=self.args.crop_size),
    #         # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         tr.ToTensor()])
    #     return composed_transforms(sample)
    
    # def transform_ts(self, sample):
    #     composed_transforms = transforms.Compose([
    #         tr.FixedResize(size=self.args.crop_size),
    #         # tr.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    #         tr.ToTensor()])
    #     return composed_transforms(sample)
    
    
#     def __init__(self,  files_names,files_root, normalize=False):
#         logging.info("ImgloaderPostdam->__init__->begin:")
#         random.seed(20201023)
#         self.img_dir = files_root
#         self.gt_dir = files_root.replace('img_jpg','gt')
#         self.normalize=normalize
#         self.file_names = files_names
#         self.data_set = []
        
#     def __len__(self):
#         return len(self.file_names)

#     def __getitem__(self, idx): 
#         filename = self.file_names[idx]
#         fileid = filename[:15]
        
#         imgfile = os.path.join(self.img_dir, filename)
#         gtfile = os.path.join(self.gt_dir, fileid+'_label.tif')
        
#         data_x = cv2.imread(imgfile,-1)
#         data_x = np.transpose(data_x,(2,0,1))
#         # imgds = gdal.Open(imgfile,gdal.GA_ReadOnly)
#         gtds = gdal.Open(gtfile,gdal.GA_ReadOnly)
#         # data_x = imgds.ReadAsArray()
#         # if self.normalize:
#         #     tmp = np.zeros(data_x.shape, dtype=np.float32)
#         #     data_x = cv2.normalize(data_x,tmp,alpha=0, beta=1, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
#         # data_x[np.isnan(data_x)]=0
        
#         data_y = gtds.ReadAsArray()
#         tmp = np.zeros(data_y.shape)
#         tmp[data_y==1]=1
#         tmp[data_y==2]=2
#         tmp[data_y==3]=3
#         tmp[data_y==4]=4
#         tmp[data_y==5]=5
#         tmp[data_y==6]=6
# #        print(np.unique(tmp))
        
# #        data_y[np.isnan(data_y)]=0
# #        data_y=np.expand_dims(data_y,axis=0)

#         _img = torch.FloatTensor(data_x)
#         _target = torch.LongTensor(tmp)
#         sample = {'image': _img, 'label': _target}
#         # print(sample)
#         # del imgds
#         del gtds
#         return sample
    
