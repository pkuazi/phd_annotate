# from dataloaders.datasets import cityscapes, coco, combine_dbs, pascal, sbd
from dataloaders.datasets import cloud_dataset
from dataloaders.datasets import lulc_dataset
# from dataloaders.datasets.cloud_dataset import gen_tile_from_filelist
from torch.utils.data import DataLoader
import os
import random

def make_data_loader(args, **kwargs):
    if args.dataset == 'lulc':
#        batch_size=args.batch_size
        # trainval_test_split_ratio = 0.8
        train_val_split_ratio = 0.7
    
        # files_root = os.path.join('E:/ubuntu_Shared_folder/Fengyun Satellite Competition/cloud_data/', 'img/')
        # files_root = os.path.join('/data/data/cloud_tif/', 'img/')
        files_root = os.path.join('/home/zjh/AIDataset', 'img_jpg/')    # img/
        file_names = os.listdir(files_root)
        # file_names = list(filter(lambda x: ".tif" in x and x.split('_')[1].split('.')[0] == \
        #                         '0330' or '0430', file_names))  # filter() 函数用于过滤序列，过滤掉不符合条件的元素
        
        # shuffle原文件
        random.seed(20201024)
        file_names = random.sample(file_names, len(file_names))
        # 从指定序列中随机获取指定长度的片断，sample函数不会修改原有序列
    
        train_file_names = file_names[:int(len(file_names) * train_val_split_ratio)]
        val_file_names = train_file_names[int(len(train_file_names) * train_val_split_ratio):]
        # train_file_names = train_file_names[:int(len(train_file_names) * train_val_split_ratio)]
        # test_file_names = file_names[int(len(file_names) * trainval_test_split_ratio):]
        
        # 构造训练loader
        # train_tiles = gen_tile_from_filelist(files_root, train_file_names)
        train_set = lulc_dataset.Imgdataset(args, train_file_names, shuffle = False, split = 'train')    # cloud_dataset  CLOUDdataset
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)  # shuffle=False

        # 构造验证loader
        # val_tiles = gen_tile_from_filelist(files_root, val_file_names)
        val_set = lulc_dataset.Imgdataset(args, val_file_names, shuffle = False, split = 'val') 
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)
        
        # 构造测试loader
        # test_tiles = gen_tile_from_filelist(files_root, test_file_names)
        # test_set = lulc_dataset.Imgdataset(args, test_file_names, shuffle = False, split = 'test')
        # test_loader = DataLoader(test_set,batch_size=args.batch_size, shuffle=False, **kwargs)
        
        return train_loader, val_loader, num_class
    elif args.dataset == 'cloud':
#        batch_size=args.batch_size
        trainval_test_split_ratio = 0.8
        train_val_split_ratio = 0.7

        # files_root = os.path.join('E:/ubuntu_Shared_folder/Fengyun Satellite Competition/cloud_data/', 'img/')
        # files_root = os.path.join('/data/data/cloud_tif/', 'img/')
        files_root = os.path.join('/home/wzj/cloud_data/', 'img')    # img/
        file_names = os.listdir(files_root)
        # file_names = list(filter(lambda x: ".tif" in x and x.split('_')[1].split('.')[0] == \
        #                         '0330' or '0430', file_names))  # filter() 函数用于过滤序列，过滤掉不符合条件的元素

        # shuffle原文件
        random.seed(20201024)
        file_names = random.sample(file_names, len(file_names))
        # 从指定序列中随机获取指定长度的片断，sample函数不会修改原有序列

        train_file_names = file_names[:int(len(file_names) * trainval_test_split_ratio)]
        val_file_names = train_file_names[int(len(train_file_names) * train_val_split_ratio):]
        train_file_names = train_file_names[:int(len(train_file_names) * train_val_split_ratio)]
        test_file_names = file_names[int(len(file_names) * trainval_test_split_ratio):]

        # 构造训练loader
        # train_tiles = gen_tile_from_filelist(files_root, train_file_names)
        train_set = cloud_dataset.CLOUDdataset(args, train_file_names, shuffle = False, split = 'train')    # cloud_dataset  CLOUDdataset
        num_class = train_set.NUM_CLASSES
        train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, **kwargs)  # shuffle=False

        # 构造验证loader
        # val_tiles = gen_tile_from_filelist(files_root, val_file_names)
        val_set = cloud_dataset.CLOUDdataset(args, val_file_names, shuffle = False, split = 'val')
        val_loader = DataLoader(val_set, batch_size=args.batch_size, shuffle=False, **kwargs)

        # 构造测试loader
        # test_tiles = gen_tile_from_filelist(files_root, test_file_names)
        test_set = cloud_dataset.CLOUDdataset(args, test_file_names, shuffle = False, split = 'test')
        test_loader = DataLoader(test_set,batch_size=args.batch_size, shuffle=False, **kwargs)

        return train_loader, val_loader, test_loader, num_class    
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
