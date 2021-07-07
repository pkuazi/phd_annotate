import os, sys, string

import gdal
import numpy as np
import cv2


import math




def resampling(raster, iconfile, iconSize):
    """
    影像重采样
    :param source_file: 源文件
    :param target_file: 输出影像
    :param scale: 像元缩放比例
    :return:
    """
    dataset = gdal.Open(raster)
    if dataset is None:
        print('Failed to open file:',raster)
        return
    band_count = dataset.RasterCount  # 波段数
 
    xSize = dataset.RasterXSize  # 列数
    ySize = dataset.RasterYSize  # 行数
    if xSize >= ySize:
        maxSize = xSize
    else:
        maxSize = ySize
    
    scale=float(iconSize) / float(maxSize)
    if band_count == 0 or not scale > 0:
        print("参数异常")
        return
    
    cols = int(xSize * scale)  # 计算新的行列数
    rows = int(ySize * scale)
    
    band1=dataset.GetRasterBand(1).ReadAsArray(buf_xsize=cols, buf_ysize=rows)
    band2=dataset.GetRasterBand(2).ReadAsArray(buf_xsize=cols, buf_ysize=rows)
    band3=dataset.GetRasterBand(3).ReadAsArray(buf_xsize=cols, buf_ysize=rows)
        
    imgcolor = np.dstack([band1,band2,band3])
    cv2.imwrite(iconfile,imgcolor)
    dataset=None
    return

def main(root_dir, icon_type, maxsize):
    out_path = root_dir.replace('data',icon_type)
    if not os.path.exists(out_path):
        os.makedirs(out_path)
    file_list = list(filter(lambda x: x.endswith('.tif'), os.listdir(root_dir)))
    for file in file_list:
        filename = file.split('.tif')[0]
        root_file = os.path.join(root_dir, file)
        icon_file = os.path.join(out_path, filename + ".png")
        resampling(root_file, icon_file, maxsize)
#         createPngIcon(root_file, icon_file, maxsize)
        print("Done!")
        
    
if __name__ == '__main__':
    root_dir = '/mnt/rsimages/gaofen_Jingjinji_2m/data'
    main(root_dir,'browse',1024)
    main(root_dir,'scale',128)
