# -*- coding: utf-8 -*-
"""
Created on Sun Jan  3 05:34:57 2021

@author: ink
"""
import os
import rasterio
# data_dir = 'E:/DATA/ISPRS_BENCHMARK_DATASETS/Potsdam/'
# data_dir = '/home/zjh/ISPRS_BENCHMARK_DATASETS/'
data_dir = '/mnt/rsimages/ISPRS_BENCHMARK_DATASETS/Potsdam'

img_root = os.path.join(data_dir,'2_Ortho_RGB/2_Ortho_RGB')
file_names = list(filter(lambda x: x.endswith(".tif"), os.listdir(img_root)))

for img in file_names:
    r=img.split('_')[2]
    c=img.split('_')[3]

    img = os.path.join(data_dir,'2_Ortho_RGB/2_Ortho_RGB/top_potsdam_%s_%s_RGB.tif'%(r,c))
    raster = rasterio.open(img, 'r')
    bbox = raster.bounds
    xmin=bbox.left
    ymin =bbox.bottom
    xmax =bbox.right
    ymax =bbox.top
    
    cutline = os.path.join(data_dir,'7_Bound/top_potsdam_%s_%s_bound.shp'%(r,c))
    
    inputfile =  os.path.join(data_dir,'8_OSM_buildings/osm_buildings_project.tif')
    dstfile = os.path.join(data_dir,'8_OSM_buildings/osm_buildings_%s_%s.tif'%(r,c))
    cmd = 'gdalwarp -cutline %s -te %s %s %s %s -of GTiff %s %s'%(cutline, xmin, ymin, xmax, ymax, inputfile, dstfile)
    os.system(cmd)