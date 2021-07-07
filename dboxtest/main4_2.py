# -*- coding: utf-8 -*-
"""
Created on Fri Mar 12 17:15:15 2021

@author: ink
"""
import dboxio
import os
from datetime import datetime
from multiprocessing import Pool
import osr,gdal
import time

TILE_SIZE=2000
file_root = '/home/zjh/phd_test_data/dbox'
img_path='/home/zjh/phd_test_data/tif'
def filter_dbox_tiles(filename):
    file = os.path.join(file_root, filename)
    ds = dboxio.Open(file)
    xtile_offs=ds.xtiles
    # (1000, 3000, 5000)
    ytile_offs=ds.ytiles
    # (2000, 4000, 5000)
    xtile_ids = list(range(len(xtile_offs)))
    ytile_ids = list(range(len(ytile_offs)))
    tile_ids=ds.GetTiles()
    # [[0, 0], [1, 0], [2, 0], [0, 1], [1, 1], [2, 1], [0, 2], [1, 2], [2, 2]]
    
    xoffs=set(xtile_offs)
    yoffs=set(ytile_offs)
    
    if xtile_offs[0]==TILE_SIZE:
        xoffs.add(0)
    else:
        xtile_ids.remove(0)
    if ytile_offs[0]==TILE_SIZE:
        yoffs.add(0)
    else:
        ytile_ids.remove(0)
        
    if xtile_offs[-1]-xtile_offs[-2]<TILE_SIZE:
        xoffs.remove(xtile_offs[-1])
        xoffs.remove(xtile_offs[-2])
        xtile_ids.remove(max(xtile_ids))
    elif xtile_offs[-1]-xtile_offs[-2]==TILE_SIZE:
        xoffs.remove(xtile_offs[-1])
        
    
    if ytile_offs[-1]-ytile_offs[-2]<TILE_SIZE:
        yoffs.remove(ytile_offs[-1])
        yoffs.remove(ytile_offs[-2])
        ytile_ids.remove(max(ytile_ids))
    elif ytile_offs[-1]-ytile_offs[-2]==TILE_SIZE:
        yoffs.remove(ytile_offs[-1]) 
    
    file_tile_list=[]
    for xoff in xoffs:
        for yoff in yoffs:
            file_tile_list.append((filename.replace('.DBOX','.tif'), xoff, yoff))   
    # print(file_tile_list)
    dbox_tile_list=[]
    for xtileid in xtile_ids:
        for ytileid in ytile_ids:
            dbox_tile_list.append((filename, xtileid, ytileid))
    # print(dbox_tile_list)
    return file_tile_list,dbox_tile_list


def dbox_read_tile(data):
    filename, xtileid, ytileid = data
    dboxfile = os.path.join(file_root, filename)
    ds=dboxio.Open(dboxfile)
    arr = ds.ReadTile(xtileid,ytileid)
    del ds

def tif_read(data):
    # -----------------reading-----------------
    filename,xoff, yoff = data
    tiffile = os.path.join(img_path, filename.replace('.DBOX','.tif'))
    ds = gdal.Open(tiffile)
    arr = ds.ReadAsArray(xoff, yoff, TILE_SIZE, TILE_SIZE)
    del ds

if __name__ == '__main__':
    file_names=os.listdir(file_root)
    for filename in file_names:
        if filename.endswith('.DBOX'):
            file_tile_list,dbox_tile_list=filter_dbox_tiles(filename)
            print(filename,len(file_tile_list))
            for parralnum in [2,4,8]:
                '''-----------------dbox parallel reading------------------------'''
                T1 = time.perf_counter()
                # print('[%s] Start test.' % datetime.now())
                # st=datetime.now()
                with Pool(parralnum) as p:
                    p.map(dbox_read_tile, dbox_tile_list)
                # ed = datetime.now()
                # print('[%s] Finished test.' % datetime.now())
                T2 =time.perf_counter()
                deltat1=((T2 - T1)*1000)/len(dbox_tile_list)
                print('dbox parallel %s reading %s time:'%(parralnum, filename),deltat1)
                # print('dbox parallel reading time:',ed-st)
                
                '''-----------------gdal parallel reading------------------------'''
                T1 = time.perf_counter()
                with Pool(parralnum) as p:
                    p.map(tif_read, file_tile_list)
                T2 =time.perf_counter()
                deltat2=((T2 - T1)*1000)/len(file_tile_list)
                print('gdal parallel %s reading %s time:'%(parralnum, filename),deltat2)
                
        
    



