# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 11:02:40 2021

@author: ink
"""

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

BLOCK_SIZE = 512

class Imgdataset(Dataset):
    
    NUM_CLASSES = 2
    
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
        
        imgfile = os.path.join(self.img_path, filename)
        gtfile = os.path.join(self.gt_path, filename)
        
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
        tmp[mask==255]=1
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

class Predataset(Dataset):
    
    NUM_CLASSES =2
    
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
        
        filename = self.file_names[idx]
        
        imgfile = os.path.join(self.img_path, filename)
        # uncertseg_annote_chicago36_4488_1848.tif
        gtfile = os.path.join(self.gt_path, 'randomseg_annote_'+filename)
        # print(imgfile,gtfile)
        imgds = gdal.Open(imgfile,gdal.GA_ReadOnly)
        gtds = gdal.Open(gtfile,gdal.GA_ReadOnly)
        if gtds is None:
            print('Failed to open file:',gtds)
            sys.exit(1)
        img_tif = imgds.ReadAsArray().astype(np.float32)
        mask = gtds.ReadAsArray()
        
        img_tif[np.isnan(img_tif)] = 0
        imax = img_tif.max()
        imin = img_tif.min()
        img_tif = (img_tif - imin)/(imax - imin)
        img_tif *= 255
               
        img = torch.from_numpy(img_tif).float()
        mask = torch.from_numpy(mask).float()
        
        # sample = {'image': img, 'label': gtds}

        return {'image': img, 'label': mask}