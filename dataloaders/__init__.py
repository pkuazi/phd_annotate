# from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
from dataloaders.datasets import Inria_dataset
from torch.utils.data import DataLoader
import os,gdal
import json
import random
import itertools

BLOCK_SIZE=512
OVERLAP_SIZE=300
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

def make_data_loader(args, **kwargs):
    if args.dataset == 'inria':
#        batch_size=args.batch_size
        trainval_test_split_ratio = 0.8
        train_val_split_ratio = 0.7
        
        trainval_test_split_ratio = 1
        train_val_split_ratio = 0.8
    
        files_root = '/home/zjh/Inria/2mAerialImageDataset/train/images'
        gt_root = '/home/zjh/Inria/2mAerialImageDataset/train/gt'
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
        
        # import time,json
        # test_file_names = json.dumps(test_file_names)
        # '''将test_file_names存入文件
        # '''
        # test_file = "/home/zjh/tmp/test_file_list_%s.txt"%( time.ctime().split(' ')[4]+'-'+time.ctime().split(' ')[1]+'-'+time.ctime().split(' ')[2]+'-'+time.ctime().split(' ')[3])
        # a = open(test_file, 'w')
        # a.write(test_file_names)
        # a.close()
        
        # train_tiles_list='/home/zjh/phd_annotate/run/train_file_list.txt'
        # val_tiles_list = '/home/zjh/phd_annotate/run/val_file_list.txt'
        # test_tiles_list = '/home/zjh/phd_annotate/run/test_file_list.txt'

        # b = open(train_tiles_list, "r",encoding='UTF-8')
        # out_train = b.read()
        # train_file_names = json.loads(out_train)
        
        # c = open(val_tiles_list, "r",encoding='UTF-8')
        # out_val = c.read()
        # val_file_names = json.loads(out_val)
        
        # d = open(test_tiles_list, "r",encoding='UTF-8')
        # out_test = d.read()
        # test_file_names = json.loads(out_test)
        
        # 构造训练loader
        train_set = Inria_dataset.Imgdataset(args, train_file_names,files_root,gt_root,BLOCK_SIZE, shuffle = False, split = 'train')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)  # shuffle=False

        # 构造验证loader
        val_set = Inria_dataset.Imgdataset(args, val_file_names,files_root,gt_root,BLOCK_SIZE, shuffle = False, split = 'val') 
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        
        # 构造测试loader
        test_set = Inria_dataset.Imgdataset(args, test_file_names,files_root,gt_root,BLOCK_SIZE, shuffle = False, split = 'test')
        test_loader = DataLoader(test_set,batch_size=args.batch_size, shuffle=False, **kwargs)
        
        return train_loader, val_loader, test_loader, num_class
    if args.dataset == 'inria_chicago_optimize':
        # test_files='/home/zjh/phd_annotate/run/test_file_list.txt'
        
        files_root = '/home/zjh/Inria/AerialImageDataset/chicago_tiles/tif'        
        gt_root = '/home/zjh/tmp/unet_predict'
        
        # b = open(test_files, "r",encoding='UTF-8')
        # out = b.read()
        # out =  json.loads(out)
        
        tile_files=os.listdir(files_root)
        
        files_num = len(tile_files)
        train_file_names=tile_files[:int(files_num*0.8)]
        val_file_names = tile_files[int(files_num*0.8):]
                
         # 构造训练loader
        train_set = Inria_dataset.Predataset(args, train_file_names,files_root,gt_root,BLOCK_SIZE, shuffle = False, split = 'train')
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)  # shuffle=False
        
         # 构造验证loader
        val_set = Inria_dataset.Predataset(args, val_file_names,files_root,gt_root,BLOCK_SIZE, shuffle = False, split = 'val') 
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        
        test_loader=None
        
        return train_loader, val_loader, test_loader, num_class
        
        
    if args.dataset == 'cloud':
#        batch_size=args.batch_size
        trainval_test_split_ratio = 0.8
        train_val_split_ratio = 0.7
    
        # files_root = os.path.join('E:/ubuntu_Shared_folder/Fengyun Satellite Competition/cloud_jpg/', 'img_jpg/')
        files_root = os.path.join('/data/data/cloud_tif/', 'img/')
        file_names = os.listdir(files_root)
        file_names = list(filter(lambda x: ".tif" in x and x.split('_')[1].split('.')[0] == \
                                 '0330' or '0430', file_names))  # filter() 函数用于过滤序列，过滤掉不符合条件的元素
        
        # shuffle原文件
        random.seed(20201024)
        file_names = random.sample(file_names, len(file_names))
        # 从指定序列中随机获取指定长度的片断，sample函数不会修改原有序列
    
        train_file_names = file_names[:int(len(file_names) * trainval_test_split_ratio)]
        val_file_names = train_file_names[int(len(train_file_names) * train_val_split_ratio):]
        train_file_names = train_file_names[:int(len(train_file_names) * train_val_split_ratio)]
        test_file_names = file_names[int(len(file_names) * trainval_test_split_ratio):]
        
        # # 构造训练loader
        # train_tiles = gen_tile_from_filelist(files_root, train_file_names)
        # train_set = cloud_dataset.CLOUDdataset(args, train_tiles, shuffle = False, split = 'train')
        # num_class = train_set.NUM_CLASSES
        # train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)  # shuffle=False

        # # 构造验证loader
        # val_tiles = gen_tile_from_filelist(files_root, val_file_names)
        # val_set = cloud_dataset.CLOUDdataset(args, val_tiles, shuffle = False, split = 'val') 
        # val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        
        # # 构造测试loader
        # test_tiles = gen_tile_from_filelist(files_root, test_file_names)
        # test_set = cloud_dataset.CLOUDdataset(args, test_tiles, shuffle = False, split = 'test')
        # test_loader = DataLoader(test_set,batch_size=args.batch_size, shuffle=False, **kwargs)
        
        # return train_loader, val_loader, test_loader, num_class
    
#     if args.dataset == 'pascal':
#         train_set = pascal.VOCSegmentation(args, split='train')
#         val_set = pascal.VOCSegmentation(args, split='val')
#         if args.use_sbd:
#             sbd_train = sbd.SBDSegmentation(args, split=['train', 'val'])
#             train_set = combine_dbs.CombineDBs([train_set, sbd_train], excluded=[val_set])
# 
#         num_class = train_set.NUM_CLASSES
#         train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
#         val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
#         test_loader = None
# 
#         return train_loader, val_loader, test_loader, num_class
# 
#     elif args.dataset == 'cityscapes':
#         train_set = cityscapes.CityscapesSegmentation(args, split='train')
#         val_set = cityscapes.CityscapesSegmentation(args, split='val')
#         test_set = cityscapes.CityscapesSegmentation(args, split='test')
#         num_class = train_set.NUM_CLASSES
#         train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
#         val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
#         test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, **kwargs)
# 
#         return train_loader, val_loader, test_loader, num_class
# 
#     elif args.dataset == 'coco':
#         train_set = coco.COCOSegmentation(args, split='train')
#         val_set = coco.COCOSegmentation(args, split='val')
#         num_class = train_set.NUM_CLASSES
#         train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)
#         val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
#         test_loader = None
#         return train_loader, val_loader, test_loader, num_class
    
    
    else:
        raise NotImplementedError