import sys
sys.path.append('/home/zjh/lulc_dl')

import torch
import torch.nn as nn
import torch.nn.functional as F
from modeling.sync_batchnorm.batchnorm import SynchronizedBatchNorm2d
from modeling.aspp import build_aspp
from modeling.decoder import build_decoder
from modeling.backbone import build_backbone

import os,gdal,sys
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
class DeepLab(nn.Module):
    def __init__(self, backbone='xception', output_stride=16, num_classes=6,
                 sync_bn=True, freeze_bn=False,pretrained=False):
        super(DeepLab, self).__init__()
        if backbone == 'drn':
            output_stride = 8

        if sync_bn == True:
            BatchNorm = SynchronizedBatchNorm2d
        else:
            BatchNorm = nn.BatchNorm2d

        self.backbone = build_backbone(backbone, output_stride, BatchNorm,pretrained=False)
        self.aspp = build_aspp(backbone, output_stride, BatchNorm)
        self.decoder = build_decoder(num_classes, backbone, BatchNorm)

        self.freeze_bn = freeze_bn

    def forward(self, input):
        x, low_level_feat = self.backbone(input)
        x = self.aspp(x)
        x = self.decoder(x, low_level_feat)
        x = F.interpolate(x, size=input.size()[2:], mode='bilinear', align_corners=True)
        return x

    def freeze_bn(self):
        for m in self.modules():
            if isinstance(m, SynchronizedBatchNorm2d):
                m.eval()
            elif isinstance(m, nn.BatchNorm2d):
                m.eval()

    def get_1x_lr_params(self):
        modules = [self.backbone]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

    def get_10x_lr_params(self):
        modules = [self.aspp, self.decoder]
        for i in range(len(modules)):
            for m in modules[i].named_modules():
                if self.freeze_bn:
                    if isinstance(m[1], nn.Conv2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p
                else:
                    if isinstance(m[1], nn.Conv2d) or isinstance(m[1], SynchronizedBatchNorm2d) \
                            or isinstance(m[1], nn.BatchNorm2d):
                        for p in m[1].parameters():
                            if p.requires_grad:
                                yield p

def featuremap(input,rastername, gt, proj):
    import torch
    import numpy as np
    import cv2
    from PIL import Image
    
    # ds = gdal.Open(rasterfile)
    # img = ds.ReadAsArray(0,0,512,512)
    # ysize=ds.RasterYSize
    # xsize=ds.RasterXSize
    # out_gt=ds.GetGeoTransform()
    # out_proj=ds.GetProjectionRef()
    # # img = ds.ReadAsArray(50,60,256,256)
    # img = cv2.normalize(img,img,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)

    # img = np.array(img).astype(np.float32)
    # # img /= 255
    # # print(img.shape)
    # imgx = img.transpose((1, 2, 0))
    
    # # output_path = 'E:/tmp/ft_from_layer/'
    # # cv2.imwrite(output_path + 'img.jpg', imgx)
    # # output_path = 'E:/tmp/ft_from_layer/'
    output_path = '/home/zjh/tmp/ft_from_layers/'
    # cv2.imwrite(output_path + 'img.jpg', imgx)
    
    # input = torch.from_numpy(img).float()
    # input = torch.unsqueeze(input, 0)
    input = input.cuda()
    # print(input.shape)
    model = DeepLab( backbone='xception', output_stride=8, num_classes=6,
                 sync_bn=True, freeze_bn=False,pretrained=True)
    model =model.cuda()
    model.eval()
    x, low_level_feat = model.backbone(input)
    print('backbone output size:',x.shape)
    print('backbone low_level_feat:',low_level_feat.shape)
    ftx = torch.squeeze(x, 0).detach().cpu().numpy()[:3].transpose((1, 2, 0))
    cv2.imwrite(output_path + '%s_ftx.jpg'%(rastername), ftx)
    # print(x.detach().cpu().numpy().shape)
    
    ftlow_level_feat = torch.squeeze(low_level_feat, 0).detach().cpu().numpy()[:3].transpose((1, 2, 0))
    cv2.imwrite(output_path + '%s_ftlow_level_feat.jpg'%(rastername), ftlow_level_feat)
    # print(low_level_feat.detach().cpu().numpy().shape)
    
    staspp = model.aspp(x)
    print('aspp output size:',staspp.shape)
    ftaspp = torch.squeeze(staspp, 0).detach().cpu().numpy()[:3].transpose((1, 2, 0))
    cv2.imwrite(output_path + '%s_ftaspp.jpg'%(rastername), ftaspp)
    print(staspp.detach().cpu().numpy().shape)
    
    stdecoder = model.decoder(staspp, low_level_feat)
    print('decoder output size:',stdecoder.shape)
    ftdecoder = torch.squeeze(stdecoder, 0).detach().cpu().numpy()[:3].transpose((1, 2, 0))
    cv2.imwrite(output_path + '%s_ftdecoder.jpg'%(rastername), ftdecoder)
    # print(stdecoder.detach().cpu().numpy().shape)
        
    stinterp = F.interpolate(stdecoder, size=input.size()[2:], mode='bilinear', align_corners=True)
    print('interpolate size,also the final outpu:',stinterp.shape)
    _, predict = torch.max(stinterp.data, 1)
    pred = predict[0].cpu()
    pred = pred.numpy()
    # print(pred)
    # stinterp = stinterp.cpu()
    ftinterp = torch.squeeze(stinterp, 0).detach().cpu().numpy()
    
    # print('before argmax:',stinterp)
    # outputs = np.argmax(ftinterp, axis=1)
    # print('after argmax:',outputs)
    # sys.exit()
    # ftinterp = torch.squeeze(stinterp, 0).detach().cpu().numpy()[:3].transpose((1, 2, 0))
    # cv2.imwrite(output_path + 'ftinterp.jpg', ftinterp)
    # print(stinterp.detach().cpu().numpy().shape)
    # # return ftinterp
    # return low_level_feat, x, staspp,stinterp
    dst_nbands = ftinterp.shape[0]
    import gdal
    dst_format = 'GTiff'
    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(os.path.join(output_path,'deeplab_lastlayer.tif'), 512, 512, dst_nbands, 6,['COMPRESS=LZW'])
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)
    
    if dst_nbands == 1:
        dst_ds.GetRasterBand(1).WriteArray(ftinterp)
    else:
        for i in range(dst_nbands):
            print(i)
            dst_ds.GetRasterBand(i + 1).WriteArray(ftinterp[i, :, :])
    del dst_ds

import itertools
# itertools模块提供的全部是处理迭代功能的函数,它们的返回值不是list,而是迭代对象,只有用for循环迭代的时候才真正计算
def gen_tiles_offs(xsize, ysize, BLOCK_SIZE=512,OVERLAP_SIZE=0):
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
    # print(xoff_list, yoff_list)
    return [d for d in itertools.product(xoff_list,yoff_list)]

def gen_file_list(geotif):
    file_list = []
    ds = gdal.Open(geotif)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    off_list = gen_tiles_offs(xsize, ysize, 512, 0)
   
    for xoff,yoff in off_list:
        file_list.append((geotif, xoff, yoff))
    return file_list

def main():
    import torch
    import numpy as np
    import cv2
    from PIL import Image
    
    input_path = '/home/zjh/ISPRS_BENCHMARK_DATASETS/2_Ortho_RGB/2_Ortho_RGB/top_potsdam_2_10_RGB.tif'
    # input_path = 'E:/DATA/ISPRS_BENCHMARK_DATASETS/Potsdam/2_Ortho_RGB/2_Ortho_RGB/top_potsdam_2_10_RGB.tif'
    
    rgb_dir = '/home/zjh/ISPRS_BENCHMARK_DATASETS/2_Ortho_RGB/2_Ortho_RGB'
    output_path = '/home/zjh/ISPRS_BENCHMARK_DATASETS/11_deeplab_feature'
    rsts = list(filter(lambda x: x.endswith('.tif') and x.startswith('top_potsdam_2_10'), os.listdir(rgb_dir))) 
    for rastername in rsts:
        print(rastername)
        rasterfile = os.path.join(rgb_dir,rastername)
        # rasterfile='/home/zjh/xa098_b4321.tif'
        file_dst=os.path.join(output_path, rastername.split('.tif')[0]+'_deeplabv3+.tif')
        
        dataset = gdal.Open(rasterfile)
        ysize=dataset.RasterYSize
        xsize=dataset.RasterXSize
        print(xsize,ysize)
        img = dataset.ReadAsArray()
        img = cv2.normalize(img,img,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        
        gt=dataset.GetGeoTransform()
        out_proj=dataset.GetProjectionRef()
        
        files_list = gen_file_list(rasterfile)

        pred_arr = np.zeros([6,ysize,xsize])
        # pred_arr = np.zeros([512,512])
        num = len(files_list)
        # for i in range(num):
        for i in range(1):
            print("processing:",i)
            _,xoff,yoff = files_list[i]
            print(xoff, yoff)
            
            tile = np.array(img[:,yoff:yoff+512, xoff:xoff+512]).astype(np.float32)
            
            input = torch.from_numpy(tile).float()
            input = torch.unsqueeze(input, 0)
            out_gt=(gt[0]+xoff*gt[1],gt[1],gt[2],gt[3]+gt[5]*yoff, gt[4],gt[5])
            outputs= featuremap(input,rastername+str('_i'),out_gt, out_proj)
            
            
    #         tile_pred = predict.cpu().numpy()
            # pred_arr[yoff:yoff+512,xoff:xoff+512]=outputs
            # pred_arr[:,yoff:yoff+512,xoff:xoff+512]=outputs
            
    #         dst_format = 'GTiff'
    #         driver = gdal.GetDriverByName(dst_format)
    #         dst_nbands = 1
    #         # dst_nbands = 6
    #         file_dst=os.path.join(output_path, rastername.split('.tif')[0]+'_%s_deeplabv3+.tif'%i)
    #         print(file_dst)
    #         dst_ds = driver.Create(file_dst, 512, 512, dst_nbands, 6,['COMPRESS=LZW'])
    #         # dst_ds = driver.Create(file_dst, xsize, ysize, dst_nbands, 6,['COMPRESS=LZW'])
    #         dst_ds.SetGeoTransform(out_gt)
    #         dst_ds.SetProjection(out_proj)

    #         if dst_nbands == 1:
    #             dst_ds.GetRasterBand(1).WriteArray(outputs)
    #         else:
    #             for i in range(dst_nbands):
    #                 arr=outputs[i, :, :]
    #                 print(arr.shape)
    #                 dst_ds.GetRasterBand(i + 1).WriteArray(arr)
    #         del dst_ds
            
        # dst_nbands = 6
        # dst_format = 'GTiff'
        # driver = gdal.GetDriverByName(dst_format)
        # dst_ds = driver.Create(file_dst, 512, 512, dst_nbands, 6,['COMPRESS=LZW'])
        # # dst_ds = driver.Create(file_dst, xsize, ysize, dst_nbands, 6,['COMPRESS=LZW'])
        # dst_ds.SetGeoTransform(out_gt)
        # dst_ds.SetProjection(out_proj)
        # if dst_nbands == 1:
        #     dst_ds.GetRasterBand(1).WriteArray(pred_arr)
        # else:
        #     for i in range(dst_nbands):
        #         arr=pred_arr[i, :, :]
        #         print(arr.shape)
        #         dst_ds.GetRasterBand(i + 1).WriteArray(arr)
        # del dst_ds
        
if __name__ == "__main__":
    main()
#     model = DeepLab(backbone='xception', output_stride=16)
# #     model.eval()
# #     input = torch.rand(1, 3, 513, 513)
# #     output = model(input)
# #     print(output.size())
    
# #     import torchvision.models as models
# #     backbone = model.backbone
# #     print(backbone.conv1)
# # #     Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
# #     backbone.conv1= nn.Conv2d(4, 64, kernel_size=7, stride=2, padding=3,bias=False)
# #     print(backbone.conv1)
    
#     import torch
#     import numpy as np
#     import cv2
#     from PIL import Image
#     import os
#     # os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
    
        
#     input_path = '/home/zjh/ISPRS_BENCHMARK_DATASETS/2_Ortho_RGB/2_Ortho_RGB/top_potsdam_2_10_RGB.tif'
#     # input_path = 'E:/DATA/ISPRS_BENCHMARK_DATASETS/Potsdam/2_Ortho_RGB/2_Ortho_RGB/top_potsdam_2_10_RGB.tif'
#     import gdal
#     ds = gdal.Open(input_path)
#     img = ds.ReadAsArray()
#     # img = ds.ReadAsArray(50,60,256,256)
#     img = cv2.normalize(img,img,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
    
#     # img = Image.open(input_path).convert('RGB')
#     # xoff = 0
#     # yoff = 0
#     # # xoff = np.random.randint(0,725)
#     # # yoff = np.random.randint(0,944)
#     # img = img.crop((xoff, yoff, xoff + 981, yoff + 981))
#     # mean = (0.485, 0.456, 0.406)
#     # std = (0.229, 0.224, 0.225)

#     img = np.array(img).astype(np.float32)
#     # img /= 255
#     print(img.shape)
#     imgx = img.transpose((1, 2, 0))
    
#     # output_path = 'E:/tmp/ft_from_layer/'
#     output_path = '/home/zjh/tmp/ft_from_layers/'
#     cv2.imwrite(output_path + 'img.jpg', imgx)
    
#     input = torch.from_numpy(img).float()
#     input = torch.unsqueeze(input, 0)
#     # print(input.shape)
#     model = DeepLab( backbone='xception', output_stride=8, num_classes=6,
#                  sync_bn=True, freeze_bn=False)
#     # input = torch.rand(1, 3, 512, 512)
#     # print(model.block1)
#     model.eval()
#     x, low_level_feat = model.backbone(input)
#     staspp = model.aspp(x)
#     ftaspp = torch.squeeze(staspp, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ftaspp.jpg', ftaspp)
#     print(staspp.detach().numpy().shape)
    
#     stdecoder = model.decoder(staspp, low_level_feat)
#     ftdecoder = torch.squeeze(stdecoder, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ftdecoder.jpg', ftdecoder)
#     print(stdecoder.detach().numpy().shape)
    
#     stinterp = F.interpolate(stdecoder, size=input.size()[2:], mode='bilinear', align_corners=True)
#     ftinterp = torch.squeeze(stinterp, 0).detach().numpy()[:3].transpose((1, 2, 0))
#     cv2.imwrite(output_path + 'ftinterp.jpg', ftinterp)
#     print(stinterp.detach().numpy().shape)
    
    





