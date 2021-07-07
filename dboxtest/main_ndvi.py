# -*- coding: utf-8 -*-
"""
Created on Thu Mar 18 16:41:02 2021

@author: ink
"""
import dboxio
import os
from datetime import datetime
from multiprocessing import Pool
import osr,gdal
import time

band3 = "/home/zjh/phd_test_data/ndvi/dbox/LC08_L1GT_123032_20200819_20200823_01_T2_B4.DBOX"  
band4 = "/home/zjh/phd_test_data/ndvi/dbox/LC08_L1GT_123032_20200819_20200823_01_T2_B5.DBOX"

ds = dboxio.Open(band3)
# 采用 with 上下文环境中，自动关闭 ds，建议使用这种方式，上下文范围外，自动释放内存资源
ndvi_ds_path = '/home/zjh/phd_test_data/ndvi.DBOX'

# 依据 band3 的剖分布局，生成一个空白剖分数据，用于保存ndvi的剖分文件
new_ds = ds.CreateAsDBoxDataset(ndvi_ds_path, dboxio.GDT_Float32,-9999)

ds4=dboxio.Open(band4)

tiles = ds.GetTiles()
print(len(tiles))
def cal_ndvi(data):
    xtile_id,ytile_id=data
    print(xtile_id,ytile_id)
    arr3=ds.ReadTile(xtile_id,ytile_id)
    arr4=ds4.ReadTile(xtile_id,ytile_id)
    ndvi=(arr4-arr3)/(arr4+arr3)
    new_ds.WriteTile(xtile_id,ytile_id,ndvi)
   
    
'''-----------------dbox parallel reading------------------------'''
T1 = time.perf_counter()
with Pool(4) as p:
    p.map(cal_ndvi, tiles)
T2 =time.perf_counter()
deltat1=((T2 - T1)*1000)
print('dbox parallel %s reading %s time:'%(4, band3),deltat1)


