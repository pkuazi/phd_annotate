# -*- coding: UTF-8 -*-
import os
import numpy as np
import torch
from modeling.deeplab import *
import gc 
from modeling.UNet_SNws import *
import time
from datetime import datetime 
from dataloaders import make_data_loader
from utils.metrics import Evaluator
from sklearn.metrics import confusion_matrix
import argparse
from tqdm import tqdm
from utils.raster_io import array_to_tiff
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
# Test the trained model
def predict():
    parser = argparse.ArgumentParser(description="PyTorch Unet Predicting")  # 创建解析器
    parser.add_argument('--dataset', type=str, default='potsdam',
                        choices=['pascal', 'coco', 'cityscapes', 'potsdam','lulc'],
                        help='dataset name (default: pascal)')
    parser.add_argument('--use-sbd', action='store_true', default=True,
                        help='whether to use SBD dataset (default: True)')
    parser.add_argument('--workers', type=int, default=0,  # default=4
                        metavar='N', help='dataloader threads')
    parser.add_argument('--base-size', type=int, default=256,  # default=513
                        help='base image size')
    parser.add_argument('--sync-bn', type=bool, default=None,  # 同步
                        help='whether to use sync bn (default: auto)')
    parser.add_argument('--freeze-bn', type=bool, default=False,  # 冻结
                        help='whether to freeze bn parameters (default: False)')
    parser.add_argument('--loss-type', type=str, default='ce',
                        choices=['ce', 'focal'],
                        help='loss func type (default: ce)')
    # cuda, seed and logging
    parser.add_argument('--no-cuda', action='store_true', default=
                        False, help='disables CUDA training')
    parser.add_argument('--gpu-ids', type=str, default='0',
                        help='use which gpu to train, must be a \
                        comma-separated list of integers only (default=0)')  # comma-separated:逗号分割
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    # checking point
    parser.add_argument('--resume', type=str, default=None,
                        help='put the path to resuming file if needed')  # 恢复文件
    parser.add_argument('--checkname', type=str, default=None,
                        help='set the checkpoint name')
    # finetuning pre-trained models
    parser.add_argument('--ft', action='store_true', default=False,
                        help='finetuning on a different dataset')
                        # action='store_true':只要运行时该变量有传参就将该变量设为True
    parser.add_argument('--batch-size', type=int, default=1,
                        metavar='N', help='input batch size for \
                                testing (default: auto)')
    # evaluation option
    parser.add_argument('--eval-interval', type=int, default=1,
                        help='evaluation interval (default: 1)')
    parser.add_argument('--no-val', action='store_true', default=False,
                        help='skip validation during training')
    
    parser.add_argument('--model_name', type=str, default='unet', choices=['deeplabv3+', 'unet3+','unet'])
    parser.add_argument('--n_channels', type=int, default=4)
    parser.add_argument('--n_filters', type=int, default=64)
    parser.add_argument('--n_class', type=int, default=2)
    parser.add_argument('--using_movavg', type=int, default=1)
    parser.add_argument('--using_bn', type=int, default=1)
    args = parser.parse_args()
    args.cuda = not args.no_cuda and torch.cuda.is_available()

    kwargs = {'num_workers': 0, 'pin_memory': True}
    train_loader, val_loader, test_loader, nclass = make_data_loader(args, **kwargs)
     #predict on the test dataset
    evaluator = Evaluator(nclass)
    evaluator.reset()  # 创建全为0的混淆矩阵
    tbar = tqdm(test_loader, desc='\r')  # 回车符

    # net = UNet_SNws(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn).cuda()
    net = UNet_SNws(args.n_channels, args.n_filters, args.n_class, args.using_movavg, args.using_bn)
    # model = os.path.join(os.getcwd(), 'run/potsdam/unet/model_best.pth.tar')
    model = os.path.join(os.getcwd(),'run/potsdam/unet/experiment_27/checkpoint.pth.tar')
    print('[%s] Start test using: %s.' % (datetime.now(), model.split('/')[-1]))
    
    # if torch.cuda.is_available():
    #     checkpoint = torch.load(model)
    #     net.load_state_dict(checkpoint['state_dict'])
    #     checkpoint= None 
    # else:
    #     checkpoint = torch.load(model,map_location=torch.device('cpu'))
    #     net.load_state_dict(checkpoint['state_dict'])
    #     checkpoint= None
    
    checkpoint = torch.load(model,map_location=torch.device('cpu'))
    net.load_state_dict(checkpoint['state_dict'])
    checkpoint= None
    # Test the trained model
    print('[%s] Start test.' % datetime.now())
    test_loss = 0.0
    # start test
    net.eval()
    
    for i, sample in enumerate(tbar):
        image, target = sample['image'], sample['label']
#            image, target = sample[0], sample[1]
        # if args.cuda:
        #     image, target = image.cuda(), target.cuda()
        # with torch.no_grad():
        #     output = net(image)
        image = torch.tensor(image, dtype=torch.float32)
        output = net(image)
        # output = net(inputs)
        score, predict = torch.max(output, 1)
        
        y_pred = predict[0].numpy()
        y_true= target.cpu().numpy()
        cm = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1), labels=[0,1])
        # cm = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1), labels=[1, 2, 3, 4, 5, 6, 7, 8, 9, 255])
        print(cm)
        # cm = cm[:5,:5]
        # freq = np.sum(cm, axis=1) / np.sum(cm)
        freq = np.sum(cm[:-1], axis=1) / np.sum(cm[:-1])
        iu = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) - np.diag(cm))
        FWIoU = (freq[freq > 0] * iu[:-1][freq > 0]).sum()
        
        MIoU = np.diag(cm) / (
                    np.sum(cm, axis=1) + np.sum(cm, axis=0) -
                    np.diag(cm))
        MIoU = np.nanmean(MIoU)
        
        test_loss += MIoU
        tbar.set_description('Test miouloss: %.3f' % (test_loss / (i + 1)))
        pred = output.data.cpu().numpy()
        target = target.cpu().numpy()
        pred = np.argmax(pred, axis=1)  # 按行
        # Add batch sample into evaluator
        evaluator.add_batch(target, pred)

    # Fast test during the training
    Acc = evaluator.Pixel_Accuracy()
    Acc_class = evaluator.Pixel_Accuracy_Class()
    mIoU = evaluator.Mean_Intersection_over_Union()
    FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()  # 频权交并比
    print('test:')
    print("Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    print('Loss: %.3f' % (test_loss / len(tbar)))

def unet_predict(array,model):
#     gc.enable()
#     gc.set_debug(gc.DEBUG_UNCOLLECTABLE ) # gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_STATS | 
    
    # Define network
    # net = UNet_SNws(3, 64, 6, using_movavg=1, using_bn=1)
    
    # model = os.path.join(os.getcwd(),'run/lulc/unetv2/model_best.pth.tar')
    net = UNet_SNws(4, 64, 6, using_movavg=1, using_bn=1)
    # model = os.path.join(os.getcwd(), 'run/potsdam/unet/model_best.pth.tar')
    # model = os.path.join(os.getcwd(),'run/potsdam/unet/experiment_27/checkpoint.pth.tar')
    print('[%s] Start test using: %s.' % (datetime.now(), model.split('/')[-1]))
    
    if torch.cuda.is_available():
        checkpoint = torch.load(model)
        net.load_state_dict(checkpoint['state_dict'])
        checkpoint= None 
    else:
        checkpoint = torch.load(model,map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['state_dict'])
        checkpoint= None
    # Test the trained model
    print('[%s] Start test.' % datetime.now())
    
    # start test
    net.eval()
        
    arr = np.expand_dims(array,axis=0)
    inputs = torch.tensor(arr, dtype=torch.float32)
    
    outputs = net(inputs)  # with shape NCHW
    
    import torch.nn.functional as F
    m = nn.Softmax(dim=1)
    out_prob=m(outputs)

    return out_prob
    # del inputs
    # # print(outputs.shape)
    # score, predict = torch.max(outputs, 1)
    # del outputs
        
    # pred = predict[0].numpy()
    # # print(pred.max())
    # print('[%s] Finished test.' % datetime.now())
    
    # del net 
    # return pred
def deeplabv3_predict(array,model):
#     gc.enable()
#     gc.set_debug(gc.DEBUG_UNCOLLECTABLE ) # gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_STATS | 
    
    # Define network
    
    net = DeepLab(backbone='xception',
                            output_stride=8,num_classes=6,
                            sync_bn=None,
                            freeze_bn=False, pretrained=True)

    backbone = net.backbone
            
    backbone.conv1 = nn.Conv2d(4, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False)
    print('change the input channels', backbone.conv1) 
    
    print('[%s] Start test using: %s.' % (datetime.now(), model.split('/')[-1]))
    
    if torch.cuda.is_available():
        checkpoint = torch.load(model)
        net.load_state_dict(checkpoint['state_dict'])
        checkpoint= None 
    else:
        checkpoint = torch.load(model,map_location=torch.device('cpu'))
        net.load_state_dict(checkpoint['state_dict'])
        checkpoint= None 
        
#     print( "\nbegin collect...")
#     _unreachable = gc.collect()
#     print( "unreachable object num:%d" %(_unreachable) )
#     print( "garbage object num:%d" %(len(gc.garbage)) )
#     time.sleep(5)

    # Test the trained model
    print('[%s] Start test.' % datetime.now())
    # start test
    net.eval()
    
    arr = np.expand_dims(array,axis=0)
    inputs = torch.tensor(arr, dtype=torch.float32)
    
    outputs = net(inputs)  # with shape NCHW
    
    return outputs
    # del inputs
    
    # _, predict = torch.max(outputs.data, 1)
    # del outputs
        
    # pred = predict[0].numpy()
    # print('[%s] Finished test.' % datetime.now())
    
    # del net 
    # return pred

def count_freq(array):
    import numpy as np
    l=sorted([(np.sum(array==i),i) for i in set(array.flat)])
    '''
    np.sum(b==i) #统计b中等于i的元素个数
    set(b.flat)  #将b转为一维数组后，去除重复元素
    sorted()     #按元素个数从小到大排序
    l[-1]        #取出元素个数最多的元组对 (count,element)
    '''
    print('max times of element in b is {1} with {0} times'.format(*l[-1]))
    return l[-1]

def miou(y_true, y_pred):
    cm = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1), labels=[0,1, 2, 3, 4, 5])
    # print(cm)
    MIoU = np.diag(cm) / (np.sum(cm, axis=1) + np.sum(cm, axis=0) -np.diag(cm))
    MIoU = np.nanmean(MIoU)
    return MIoU

def main():
    from utils.gen_tiles_offs import gen_tiles_offs
    # predict()
    input_path = '/home/zjh/ISPRS_BENCHMARK_DATASETS/4_Ortho_RGBIR/4_Ortho_RGBIR/top_potsdam_2_11_RGBIR.tif'
    import gdal,cv2
    ds = gdal.Open(input_path)
    gt = ds.GetGeoTransform()
    proj = ds.GetProjectionRef()
    xsize=ds.RasterXSize
    ysize = ds.RasterYSize
    # print(xsize, ysize)
    files_list = gen_file_list(input_path)
    pred_arr = np.zeros([xsize,ysize])
    num = len(files_list)
    
    for i in range(num):
        
        _,xoff,yoff = files_list[i]
        # print(xoff, yoff)
        img_tif = ds.ReadAsArray(xoff,yoff,512,512)
        # img = cv2.normalize(img,img,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        img_tif[np.isnan(img_tif)] = 0
        imax = img_tif.max()
        imin = img_tif.min()
        img_tif = (img_tif - imin)/(imax - imin)
        img_tif *= 255
    # #     img = '/tmp/data/6405_2015_56_02_LC81290342015244LGN00.jpg'
    # #     import cv2 
    # #     array = cv2.imread(img,-1)
    # #     array = np.transpose(array,(2,0,1))
        pred_ = unet_predict(img_tif)
        
#         tile_pred = predict.cpu().numpy()
        pred_arr[yoff:yoff+512,xoff:xoff+512]=pred_
    
    # dst_file = '/tmp/clt2.tif'
    dst_file = '/home/zjh/tmp/top_osm_2_11.tif'

    dst_format = 'GTiff'
    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(dst_file, ysize, xsize, 1, 1)
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)
    dst_ds.GetRasterBand(1).WriteArray(pred_arr)

if __name__ == '__main__':
    # main()
    import pandas as pd
    import cv2
    import json
    tile_size=512
    # model_path='/home/zjh/lulc_dl/run/potsdam/unet/experiment_59'
    model_path='/home/zjh/lulc_dl/run/potsdam/deeplab-xception/experiment_2'
    # test_files=os.path.join(model_path, 'test_file_list_2021-Jan-14-04:50:30.txt')
    test_files='/home/zjh/lulc_dl/run/test_file_list_2021-Jan-18-14:59:55.txt'
    # model=os.path.join(model_path,'checkpoint.pth.tar')
    model = '/home/zjh/lulc_dl/run/potsdam/unet/model_best.pth.tar'
    data_path = '/home/zjh/ISPRS_BENCHMARK_DATASETS/4_Ortho_RGBIR/4_Ortho_RGBIR'
    label_path = '/home/zjh/ISPRS_BENCHMARK_DATASETS/9_labels_id/'
    b = open(test_files, "r",encoding='UTF-8')
    out = b.read()
    out =  json.loads(out)
    
    result_path='/home/zjh/tmp/unet_predict'
    score=0
    i=0
    max_score=0
    max_file=''
    for file in out:
        i=i+1
        filename, xoff, yoff = file
        ds = gdal.Open(os.path.join(data_path,filename))
        
        out_proj=ds.GetProjection()
        gt = ds.GetGeoTransform()
        out_gt = list(gt)
        out_gt[0] = gt[0] + xoff * gt[1]
        out_gt[3] = gt[3] + yoff * gt[5]
        
        img_tif = ds.ReadAsArray(xoff, yoff, tile_size,tile_size)
        # img = cv2.normalize(img,img,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        img_tif[np.isnan(img_tif)] = 0
        imax = img_tif.max()
        imin = img_tif.min()
        img_tif = (img_tif - imin)/(imax - imin)
        img_tif *= 255
        
        outputs = unet_predict(img_tif,model)
        # outputs = deeplabv3_predict(img_tif,model)
        
        outputarr = torch.squeeze(outputs, 0).detach().cpu().numpy()
        print(outputarr.shape)
        print(outputarr[:,10,10])
        array_to_tiff(outputarr,out_proj, out_gt, outputarr.shape[0], os.path.join(result_path,'softmax_prob_%s_%s_%s.tif'%(filename.split('.')[0],xoff, yoff)))
    #     score, predict = torch.max(outputs, 1)
    #     pred_ = predict[0].numpy()
    #     # print(pred.max())
    #     print('[%s] Finished test.' % datetime.now())
    #     array_to_tiff(pred_, out_proj, out_gt, 1, os.path.join(result_path,'pred_%s_%s_%s.tif'%(filename.split('.')[0],xoff, yoff)))
    #     # cv2.imwrite(os.path.join(result_path,'pred_%s_%s_%s.tif'%(filename.split('.')[0],xoff, yoff)),pred_)
        
    #     # # cm = confusion_matrix(y_true.reshape(-1), y_pred.reshape(-1), labels=[1, 3, 6, 7, 9, 255])
    #     label = os.path.join(label_path,filename.replace('RGBIR','label'))
    #     ds_l=gdal.Open(label)
    #     y_true=ds_l.ReadAsArray(xoff, yoff, tile_size, tile_size)
    #     # cv2.imwrite(os.path.join(result_path,'label_%s_%s_%s.tif'%(filename.split('.')[0],xoff, yoff)),y_true)
        
    #     fasle_pred = y_true==pred_
    #     fasle_pred=fasle_pred.astype('int8')
    #     array_to_tiff(fasle_pred, out_proj, out_gt, 1, os.path.join(result_path,'falsepred_%s_%s_%s.tif'%(filename.split('.')[0],xoff, yoff)))
        
    #     preMIoU=miou(y_true, pred_)
    #     if preMIoU>max_score:
    #         max_score=preMIoU
    #         max_file=file
    #     print('pre:',preMIoU)
    #     score+=preMIoU
        
    # print('final miou,',score/i)
    # print('max score of %s is %s'%(max_file, max_score))
    
    # df = pd.read_csv('/home/zjh/tmp/deeplab_seg/ergctop_potsdam_4_13_2560_5120.csv',sep=',',header=None)
    # segments_ergc=df.values
    # post_pred = np.zeros([512,512])
    # for i in range(segments_ergc.max()+1):
    #     segment = pred_[segments_ergc==i]
    #     f,value=count_freq(segment)
    #     post_pred[segments_ergc==i]=value
    # cv2.imwrite('/home/zjh/tmp/deeplab_seg/post_pred_top_potsdam_4_13_2560_5120.tif',post_pred)
    
