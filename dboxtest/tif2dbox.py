# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 11:13:01 2021

@author: ink
"""
# 数据格式转换tif to dbox
# /home/zjh/Inria/AerialImageDataset/train/images

# dboxdataset copy chicago10.tif /home/zjh/Inria/AerialImageDataset/train/dbox/chicago10.DBOX
import os

img_path = '/home/zjh/Inria/AerialImageDataset/train/images'
dbox_path = '/home/zjh/Inria/AerialImageDataset/train/dbox'

tifs = os.listdir(img_path)
for file in tifs:
    if '1' in file or '2' in file or '3' in file:
        continue
    tiffile=os.path.join(img_path,file)
    dboxfile = os.path.join(dbox_path, file.replace('.tif', '.DBOX'))
    cmd = 'dboxdataset copy %s %s'%(tiffile,dboxfile)
    os.system(cmd)