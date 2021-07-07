# -*- coding: utf-8 -*-
"""
Created on Wed Jul  7 11:01:34 2021

@author: zjh
"""

import os

# def resampling(input_im, output_im):
#     cmd = 'gdalwarp -ts 750 750 -r bilinear %s %s '%('chicago10.tif','/home/zjh/Inria/2mAerialImageDataset/train/chicago10_2m.tif' )
#     os.system(cmd)

input_path='/home/zjh/Inria/AerialImageDataset/train/gt'
output_path='/home/zjh/Inria/2mAerialImageDataset/train/gt'

inputfiles=os.listdir(input_path)
for filename in inputfiles:
    input_im = os.path.join(input_path, filename)
    output_im=os.path.join(output_path, filename.split('.')[0]+'_2m.'+filename.split('.')[1])
    cmd = 'gdalwarp -ts 750 750 -r bilinear %s %s '%(input_im,output_im )
    os.system(cmd)