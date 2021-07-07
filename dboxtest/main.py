# -*- coding: utf-8 -*-
"""
Created on Wed Mar 17 09:37:59 2021

@author: ink
"""
import dboxio
import os
import pandas as pd
from datetime import datetime
from multiprocessing import Pool
import osr
import gdal
import json
import time,random

img_path='/home/zjh/phd_test_data/tif'
dbox_path='/home/zjh/phd_test_data/dbox'
files_root='/home/zjh/phd_test_data/tif'
file_names=os.listdir(files_root)



def tif_read(data):
    # -----------------reading-----------------
    filename,xoff, yoff = data
    tiffile = os.path.join(img_path, filename.replace('.DBOX','.tif'))
    
    ds = gdal.Open(tiffile)
    arr = ds.ReadAsArray(xoff, yoff, 256, 256)
    del ds

# dbox并行读
def dbox_read(data):
    filename,xoff, yoff = data
    dboxfile = os.path.join(dbox_path, filename.replace('.tif','.DBOX'))
    ds=dboxio.Open(dboxfile)
    gt = ds.GetGeoTransform()
    newgt=list(gt)
    newgt[0]=newgt[0]+xoff*newgt[1]
    newgt[3]=newgt[3]+yoff*newgt[5]
    arr = ds.ReadRegion(newgt, 256,256)

    del ds

def dbox_read_tile(data):
    xtile, ytile = data
    dboxfile = os.path.join(dbox_path, 'chicago1.DBOX')
      
    ds=dboxio.Open(dboxfile)

    arr = ds.ReadTile(xtile,ytile)

    del ds

   
if __name__ == '__main__':

    # split_train_val_test('Inria')
    # 10.0.90.63
    
    records = pd.DataFrame(columns = ['filename', 'tilenum', 'parralnum','gdal_tif','dbox_DBOX'])   
    
    files_offs_list=[]
    for filename in file_names:
        if filename.endswith(".tif"):
            print(filename)

            tilelist_file = "/home/zjh/phd_test_data/%s_tlist.txt"%(filename.split('.')[0])
     
            b = open(tilelist_file, "r",encoding='UTF-8')
            tilelist = b.read()
            tile_names = json.loads(tilelist)
            tile_names = random.sample(tile_names,len(tile_names))
            print('the number of tiles is,',len(tile_names))  
            
            for parralnum in [2,4,8]:
            
                '''-----------------dbox parallel reading------------------------'''
                T1 = time.perf_counter()
    
                # print('[%s] Start test.' % datetime.now())
                # st=datetime.now()
                with Pool(parralnum) as p:
                    p.map(dbox_read, tile_names)
                # ed = datetime.now()
                # print('[%s] Finished test.' % datetime.now())
                T2 =time.perf_counter()
                deltat1=((T2 - T1)*1000)/len(tile_names)
                print('dbox parallel %s reading %s time:'%(parralnum, filename),deltat1)
                
                '''-----------------gdal parallel reading------------------------'''
                T1 = time.perf_counter()
                with Pool(parralnum) as p:
                    p.map(tif_read, tile_names)
                T2 =time.perf_counter()
                deltat2=((T2 - T1)*1000)/len(tile_names)

                print('gdal parallel %s reading %s time:'%(parralnum, filename),deltat2)
                
                '''-----------------dbox parallel reading tiles-----------------------'''
                # T1 = time.perf_counter()
                
                # dboxfile = os.path.join(dbox_path, filename.replace('.tif','.DBOX'))    
                # ds=dboxio.Open(dboxfile)
                # tile_ids=ds.GetTiles()
                # # print('[%s] Start test.' % datetime.now())
                # # st=datetime.now()
                # with Pool(parralnum) as p:
                #     p.map(dbox_read, tile_ids)
                # # ed = datetime.now()
                # # print('[%s] Finished test.' % datetime.now())
                # T2 =time.perf_counter()
                # deltat1=((T2 - T1)*1000)/len(tile_names)
                # print(deltat1)
                # # print('dbox parallel reading time:',ed-st)
                
                new=pd.DataFrame({'filename':filename, 'tilenum':len(tile_names), 'parralnum':parralnum,'gdal_tif':deltat2,'dbox_DBOX':deltat1},index = [0])
                records=records.append(new,ignore_index=True) 
    records.to_csv(os.path.join(os.getcwd(),'test_results256.csv'))