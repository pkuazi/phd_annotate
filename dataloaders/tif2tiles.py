# -*- coding: utf-8 -*-
"""
Created on Wed Mar  3 16:52:40 2021

@author: ink
"""
import gdal
import os
import numpy as np
import random
import json
import cv2
from PIL import Image

NP2GDAL_CONVERSION = {
  "uint8": 1,
  "int8": 1,
  "uint16": 2,
  "int16": 3,
  "uint32": 4,
  "int32": 5,
  "float32": 6,
  "float64": 7,
  "complex64": 10,
  "complex128": 11,
}
def img2tiles():
    random_select_ratio = 0.1
    test_tiles_list = '/home/zjh/phd_annotate/run/test_file_list.txt'
    files_root = '/home/zjh/Inria/AerialImageDataset/train/images'
    b = open(test_tiles_list, "r",encoding='UTF-8')
    out_test = b.read()
    file_names = json.loads(out_test) 
    
    # # shuffle原文件       
    random.seed(20210303)
    test_file_names = random.sample(file_names, len(file_names))
    # # 从指定序列中随机获取指定长度的片断，sample函数不会修改原有序列
    test_file_names = test_file_names[:int(len(file_names) * random_select_ratio)]
    
    output='/home/zjh/Inria/AerialImageDataset/chicago_tiles/tif'
    png_output ='/home/zjh/Inria/AerialImageDataset/chicago_tiles/png'
    for tile in test_file_names:
        filename,xoff, yoff=tile
        imagefile=os.path.join(files_root,filename)
        # ds_img = gdal.Open(imagefile) 
        # data = ds_img.ReadAsArray(xoff,yoff, 512, 512)      
        
        
#     imagefile='/mnt/win/data/Potsdam/4_Ortho_RGBIR/top_potsdam_4_13_RGBIR.tif'
#     labelfile = '/mnt/win/data/Potsdam/9_labels_id/top_potsdam_4_13_label_proj.tif'
# #     '"top_potsdam_4_13_RGBIR.tif", 2560, 5120]'
#     xoff = 2560
#     yoff = 5120 
        xsize=512
        ysize=512
    #     imagefile='/mnt/win/data/xiongan/Jingjinji_7_15_utm.tif'
    #     labelfile = '/mnt/win/data/xiongan/N50_35_2010_jjj_7_15_resample.tif'
    #     baifile = '/mnt/win/data/xiongan/BAI/LC08_L1TP_123033_20180408_20180417_01_T1_BAI.TIF'
        ds_img = gdal.Open(imagefile) 
        out_proj=ds_img.GetProjection()
        gt = ds_img.GetGeoTransform()
        out_gt = list(gt)
        out_gt[0] = gt[0] + xoff * gt[1]
        out_gt[3] = gt[3] + yoff * gt[5]
#         ds_label = gdal.Open(labelfile)
        data = ds_img.ReadAsArray(xoff,yoff, 512, 512)
        
        pngdata = np.transpose(data,[1,2,0])
        im = Image.fromarray(pngdata)
        im = im.convert('P')
        im.save(os.path.join(png_output,'%s_%s_%s.png'%(filename.split('.')[0],xoff, yoff)))
        
        dst_nbands=data.shape[0]
        dst_format = 'GTiff'
        driver = gdal.GetDriverByName(dst_format)
        dtype = NP2GDAL_CONVERSION[str(data.dtype)]
        dst_ds = driver.Create(os.path.join(output,'%s_%s_%s.tif'%(filename.split('.')[0],xoff, yoff)), ysize, xsize, dst_nbands, dtype)
        dst_ds.SetGeoTransform(out_gt)
        dst_ds.SetProjection(out_proj)
    
        if dst_nbands == 1:
            dst_ds.GetRasterBand(1).WriteArray(data)
        else:
            for i in range(dst_nbands):
                dst_ds.GetRasterBand(i + 1).WriteArray(data[i, :, :])
        del dst_ds
def gt2tiles():
    pass
if __name__ == '__main__':   
    img2tiles()