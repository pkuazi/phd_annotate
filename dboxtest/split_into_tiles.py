# -*- coding: utf-8 -*-
"""
Created on Thu Jan 14 09:33:26 2021

@author: ink
"""
import os,gdal
import random
import itertools
import time,json
import dboxio

BLOCK_SIZE=512
OVERLAP_SIZE=0
def gen_tiles_offs(xsize, ysize, BLOCK_SIZE,OVERLAP_SIZE):
    xoff_list = []
    yoff_list = []
    
    cnum_tile = int((xsize - BLOCK_SIZE) / (BLOCK_SIZE - OVERLAP_SIZE)) + 1
    rnum_tile = int((ysize - BLOCK_SIZE) / (BLOCK_SIZE - OVERLAP_SIZE)) + 1
    
    for j in range(cnum_tile + 1):
        xoff = 0 + (BLOCK_SIZE - OVERLAP_SIZE) * j                  
        if j == cnum_tile:
            xoff = xsize - BLOCK_SIZE
        xoff_list.append(xoff)
        
    for i in range(rnum_tile + 1):
        yoff = 0 + (BLOCK_SIZE - OVERLAP_SIZE) * i
        if i == rnum_tile:
            yoff = ysize - BLOCK_SIZE
        yoff_list.append(yoff)
    
    if xoff_list[-1] == xoff_list[-2]:
        xoff_list.pop()    # pop() 方法删除字典给定键 key 及对应的值，返回值为被删除的值
    if yoff_list[-1] == yoff_list[-2]:    # the last tile overlap with the last second tile
        yoff_list.pop()
    
    return [d for d in itertools.product(xoff_list,yoff_list)]
    # itertools.product()：用于求多个可迭代对象的笛卡尔积

def gen_file_list(geotif):
    file_list = []
    filename = geotif.split('/')[-1]
    ds = gdal.Open(geotif)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    # im = Image.open(geotif).convert('RGB')
    # xsize, ysize = im.size
    off_list = gen_tiles_offs(xsize, ysize, BLOCK_SIZE, OVERLAP_SIZE)
   
    for xoff, yoff in off_list:    
        file_list.append((filename, xoff, yoff))     
    return file_list

def gen_tile_from_filelist(dir, file_names):
    files_offs_list=[]
    for filename in file_names:
        if filename.endswith(".tif"):
            file = os.path.join(dir, filename)
            tif_list = gen_file_list(file)
            files_offs_list = files_offs_list+tif_list
    return files_offs_list

def list_to_txt(listinfo, txtfile):
    listjson = json.dumps(listinfo)
    '''将test_file_names存入文件
    '''
    a = open(txtfile, 'w')
    a.write(listjson)
    a.close()
    # b = open(test_files, "r",encoding='UTF-8')
    # out = b.read()
     # out =  json.loads(out)

def split_train_val_test(dataset):
    if dataset == 'potsdam':
#        batch_size=args.batch_size
        trainval_test_split_ratio = 0.95
        train_val_split_ratio = 0.8
    
        files_root = '/home/zjh/ISPRS_BENCHMARK_DATASETS/4_Ortho_RGBIR/4_Ortho_RGBIR'
        gt_root = '/home/zjh/ISPRS_BENCHMARK_DATASETS/9_labels_id'
        # gt_root = '/home/zjh/ISPRS_BENCHMARK_DATASETS/8_OSM_buildings'
        file_names = os.listdir(files_root)
        file_names = list(filter(lambda x: x.endswith(".tif"), file_names))  # filter() 函数用于过滤序列，过滤掉不符合条件的元素
        
        file_names = gen_tile_from_filelist(files_root, file_names)
        
        # shuffle原文件
        random.seed(20201024)
        file_names = random.sample(file_names, len(file_names))
        # 从指定序列中随机获取指定长度的片断，sample函数不会修改原有序列
    
        train_file_names = file_names[:int(len(file_names) * trainval_test_split_ratio)]
        val_file_names = train_file_names[int(len(train_file_names) * train_val_split_ratio):]
        train_file_names = train_file_names[:int(len(train_file_names) * train_val_split_ratio)]
        test_file_names = file_names[int(len(file_names) * trainval_test_split_ratio):]
        
        print('train, val, test are: ',(len(train_file_names), len(val_file_names),len(test_file_names)))
        
        # timest =  time.ctime().split(' ')[4]+'-'+time.ctime().split(' ')[1]+'-'+time.ctime().split(' ')[2]+'-'+time.ctime().split(' ')[3]
        # test_file = "/home/zjh/lulc_dl/run/test_file_list_%s.txt"%(timest)
        # list_to_txt(test_file_names, test_file)
        
        # train_file = "/home/zjh/lulc_dl/run/train_file_list_%s.txt"%(timest)
        # list_to_txt(train_file_names, train_file)
        
        # val_file = "/home/zjh/lulc_dl/run/val_file_list_%s.txt"%(timest)
        # list_to_txt(val_file_names, val_file)
    if dataset == 'Inria':
#        batch_size=args.batch_size
        # trainval_test_split_ratio = 0.95
        train_val_split_ratio = 0.8
    
        files_root = '/home/zjh/Inria/AerialImageDataset/train/images'
        gt_root = '/home/zjh/Inria/AerialImageDataset/train/gt'
        # gt_root = '/home/zjh/ISPRS_BENCHMARK_DATASETS/8_OSM_buildings'
        file_names = os.listdir(files_root)
        
        #for chicago city data
        testfile_names =  list(filter(lambda x: x.endswith('.tif') and x.startswith("chicago"), file_names))  # filter() 函数用于过滤序列，过滤掉不符合条件的元素
        testfile_names =  gen_tile_from_filelist(files_root, testfile_names)
        print('test are: ',(len(testfile_names)))
        test_file = "/home/zjh/phd_annotate/run/test_file_list.txt"
        list_to_txt(testfile_names, test_file)
        return
        
        file_names = list(filter(lambda x: x.endswith('.tif') and not x.startswith("chicago"), file_names))  # filter() 函数用于过滤序列，过滤掉不符合条件的元素
        file_names = gen_tile_from_filelist(files_root, file_names)
        
        # shuffle原文件
        random.seed(20201024)
        file_names = random.sample(file_names, len(file_names))
        # 从指定序列中随机获取指定长度的片断，sample函数不会修改原有序列
    
        train_file_names = file_names[:int(len(file_names) * train_val_split_ratio)]
        val_file_names = file_names[int(len(train_file_names) * train_val_split_ratio):]
        # train_file_names = train_file_names[:int(len(train_file_names) * train_val_split_ratio)]
        # test_file_names = file_names[int(len(file_names) * trainval_test_split_ratio):]
        
        print('train, val, test are: ',(len(train_file_names), len(val_file_names)))
        
        timest =  time.ctime().split(' ')[4]+'-'+time.ctime().split(' ')[1]+'-'+time.ctime().split(' ')[2]+'-'+time.ctime().split(' ')[3]
        # test_file = "/home/zjh/lulc_dl/run/test_file_list_%s.txt"%(timest)
        # list_to_txt(test_file_names, test_file)
        
        train_file = "/home/zjh/phd_annotate/run/train_file_list_%s.txt"%(timest)
        list_to_txt(train_file_names, train_file)
        
        val_file = "/home/zjh/phd_annotate/run/val_file_list_%s.txt"%(timest)
        list_to_txt(val_file_names, val_file)
if __name__ == '__main__':
    # split_train_val_test('Inria')
    # 10.0.90.63
    files_root='/home/zjh/phd_test_data/tif'
    file_names=os.listdir(files_root)
    dbox_path='/home/zjh/phd_test_data/dbox'
    
    for filename in file_names:
        if filename.endswith(".tif"):
            files_offs_list=[]
            
            dboxfile = os.path.join(dbox_path, filename.replace('.tif','.DBOX'))
      
            ds=dboxio.Open(dboxfile)
            off_list=ds.GetTileOffsets()
            for xoff in off_list[0]:
                for yoff in off_list[1]:    
                    files_offs_list.append((filename, xoff, yoff))
            
            
            # file = os.path.join(files_root, filename)
            # tif_list = gen_file_list(file)
            # files_offs_list = files_offs_list+tif_list

            tilelist_file = "/home/zjh/phd_test_data/%s_dbox_tlist.txt"%(filename.split('.')[0])
            list_to_txt(files_offs_list, tilelist_file)