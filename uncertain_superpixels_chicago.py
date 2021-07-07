# -*- coding: utf-8 -*-
"""
Created on Wed Jan 20 09:57:22 2021

@author: ink
"""
import os,sys
import numpy as np
import gdal
import pandas as pd
from utils.raster_io import array_to_tiff
from sklearn.ensemble import RandomForestClassifier
from utils.metrics import Evaluator
from utils.geojson_process import geojson_attr_process
from modeling.UNet_SNws import *
import torch
from datetime import datetime 
import gdal, ogr, osr
import geopandas
import random

def secondmax_of_list(list1):
    # list of numbers
    # list1 = [10, 20, 4, 45, 99]
     
    # new_list is a set of list1
    new_list = set(list1)
     
    # removing the largest element from temp list
    new_list.remove(max(new_list))
     
    # elements in original list are not changed
    # print(list1)
     
    return max(new_list)
def distance_of_max_smax(probarray):
    # input probarray is 6*512*512
    class_num, rows, cols=probarray.shape
    #transform to 512*512*6
    bands = np.dstack([probarray[i] for i in range(class_num)])
    
    distarr = np.zeros((rows,cols))
    
    for x in range(cols):
        for y in range(rows):
            data = set(bands[x][y])
            firstmax = max(data)
            data.remove(max(data))
            secondmax = max(data)
            dis = firstmax-secondmax
            distarr[x][y]=dis
    return distarr
def most_frequent_in_array(b):
    import numpy as np

    # b=np.array([[0, 4, 4],[2, 0, 3],[1, 3, 4]])
    
    # print('b=')
    
    # print(b)
    
    l=sorted([(np.sum(b==i),i) for i in set(b.flat)])
    
    '''
    
    np.sum(b==i) #统计b中等于i的元素个数
    
    set(b.flat)  #将b转为一维数组后，去除重复元素
    
    sorted()     #按元素个数从小到大排序
    
    l[-1]        #取出元素个数最多的元组对 (count,element)
    
    '''
    
    # print('max times of element in b is {1} with {0} times'.format(*l[-1]))
    return l[-1][1]
def mark_uncertain_pixels(distarray,percentile):
    # input probarray is 512*512
    rows, cols=distarray.shape
    
    mark = np.zeros((rows,cols), dtype=bool)
    threshold = np.quantile(distarray, percentile)
    print('%s threshold is %s'%(percentile,threshold))
    for x in range(cols):
        for y in range(rows):
            dis = distarray[x][y]
            if dis<threshold:
                mark[x][y]=1
    return mark,threshold
def draw_histgram_of_array(arr):
    import matplotlib.pyplot as plt
    _ = plt.hist(arr, bins='auto')  # arguments are passed to np.histogram
    plt.title("Histogram with 'auto' bins")
    plt.show()
def mark_hard_to_classify_by_probarray(root_dir, distarr):
    # root_dir = 'E:/tmp/ergc_test/unet_predict'
    file_list = list(filter(lambda x: x.startswith('softmax_prob') and x.endswith('.tif') , os.listdir(root_dir)))
    for filename in file_list:
        print(filename)
        file = os.path.join(root_dir, filename)
        ds = gdal.Open(file)
        
        array = ds.ReadAsArray()
        distarray = distance_of_max_smax(array)
        draw_histgram_of_array(distarray)
        mark = mark_uncertain_pixels(distarray, distarr)#distarr=0.2
        mark=mark.astype('int8')
        
        array_to_tiff(mark, ds.GetProjection(), ds.GetGeoTransform(), 1, os.path.join(root_dir,filename.replace('prob','maxd')))

def probabilities_of_falsepred():
    root_dir = 'E:/tmp/ergc_test/unet_predict'
    file_list = list(filter(lambda x: x.startswith('falsepred') and x.endswith('.tif') , os.listdir(root_dir)))
    for filename in file_list:
        file = os.path.join(root_dir, filename)
        ds = gdal.Open(file)
        
        mask = ds.ReadAsArray()
        probf = file.replace('falsepred','prob')
        ds1=gdal.Open(probf)
        probarray=ds1.ReadAsArray()
        
        class_num, rows, cols=probarray.shape
        #transform to 512*512*6
        bands = np.dstack([probarray[i] for i in range(class_num)])
        
        tmp = np.zeros((rows,cols,3))
        for x in range(cols):
            for y in range(rows):
                if mask[x][y]==0:
                    dlist = list(bands[x][y])
                    dlist.sort(reverse=True)
                    tmp[x][y]=dlist[:3]
        tmp = np.transpose(tmp,(2,0,1))
        array_to_tiff(tmp, ds.GetProjection(), ds.GetGeoTransform(), 3, os.path.join(root_dir,filename.replace('falsepred','max3c')))
def probability_falsepred_classifier():
    root_dir = 'E:/tmp/ergc_test/unet_predict'
    file_list = list(filter(lambda x: x.startswith('prob') and x.endswith('924_1386.tif') , os.listdir(root_dir)))
    feat=[]
    train_y=[]
    for filename in file_list:
        # filename prob_top_potsdam_2_10_RGBIR_924_1386.tif
        file = os.path.join(root_dir, filename)
        ds = gdal.Open(file)
        probarray = ds.ReadAsArray()
        class_num, rows, cols=probarray.shape
        
        falsepredf=file.replace('prob','falsepred')
        fpds=gdal.Open(falsepredf)
        fparr =fpds.ReadAsArray()
        
        #transform to 512*512*6
        bands=np.transpose(probarray,(1,2,0))
        
        tmp = np.zeros((rows,cols,class_num))
        for x in range(cols):
            for y in range(rows):
                if (fparr[x][y]==1 and  np.random.rand()<0.25) or fparr[x][y]==0:
                    data = set(bands[x][y])
                    maxv = max(data)
                    tmp[x][y]=maxv-bands[x][y]
                    feat.append(tmp[x][y])
                    train_y.append(fparr[x][y])
    feat=np.array(feat)
    train_x = feat.reshape(-1, 6)

    training_X = pd.DataFrame(train_x)
    training_y = pd.DataFrame(train_y)
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=9, random_state=0)
    clf.fit(training_X, training_y)
    
    # pickle.dump(clf, open(model_npy, 'wb'))
    predf = os.path.join(root_dir,'prob_top_potsdam_2_10_RGBIR_462_1386.tif')
    predfds=gdal.Open(predf)
    nparr=predfds.ReadAsArray()
    bands=np.transpose(nparr,(1,2,0))
    feat = []
    for i in range(cols):
        for j in range(rows):
            feat.append(bands[i][j])
    feat = np.array(feat)
    df = pd.DataFrame(feat)
    
    # clf = pickle.load(open(model, 'rb'))
    r = clf.predict(df)
    classified_array = r.reshape(rows, cols)
    array_to_tiff(classified_array, predfds.GetProjection(), predfds.GetGeoTransform(), 1, os.path.join(root_dir,'rf_prob_hardc_462_1386.tif'))

def find_hard_to_classify_segments(segments_csv,hard_to_classify_f):
    seg_df=pd.read_csv(segments_csv, sep=',',header=None)
    segments_mask=seg_df.values
    
    hcds = gdal.Open(hard_to_classify_f)
    hcarr = hcds.ReadAsArray()
    hcsegments =segments_mask[hcarr==1] 
    hcsegids = set(list(hcsegments))
    print('the number of segments is ',len(hcsegids))
    list_hcsegids=list(hcsegids)
    for hcsegid in list_hcsegids:
        hcpixel_num = hcsegments[hcsegments==hcsegid].sum()
        print(hcsegid, hcpixel_num)
        segpixel_num=segments_mask[segments_mask==hcsegid].sum()
        if hcpixel_num <=0.3*segpixel_num:
            hcsegids.remove(hcsegid)
    
    hc_seg=np.full(segments_mask.shape, 9999, dtype=int)
    print('the number of segments needed to be labeled is ',len(hcsegids))
    for hcsegid in list(hcsegids):
        hc_seg[segments_mask==hcsegid]=hcsegid
    array_to_tiff(hc_seg, hcds.GetProjection(), hcds.GetGeoTransform(), 1, 'E:/tmp/ergc_test/unet_predict/hcseg_top_potsdam_2_13_RGBIR_3234_1386.tif')
    
def mark_seg_by_gt():
    hard_to_classify_f = 'E:/tmp/ergc_test/unet_predict/hcseg_top_potsdam_2_13_RGBIR_3234_1386.tif'
    
    
def main(distarr):
    root_dir='/home/zjh/tmp/unet_predict'
    segcsv = '/home/zjh/tmp/maskcsv'
    gt_dir = '/home/zjh/ISPRS_BENCHMARK_DATASETS/9_labels_id'
    # root_dir ='E:/tmp/ergc_test/unet_predict'
    # segcsv = 'E:/tmp/ergc_test/maskcsv'
    # gt_dir ='E:/DATA/ISPRS_BENCHMARK_DATASETS/Potsdam/9_labels_id'
       
    # 'softmax_prob_top_potsdam_7_13_RGBIR_462_0'
    file_list = list(filter(lambda x: x.startswith('softmax_prob') and x.endswith('.tif') , os.listdir(root_dir)))
     # log
    logfile = os.path.join(os.getcwd(),'hclog.txt')
    log_file = open(logfile, 'a')
    records = pd.DataFrame(columns = ['filename', 'threshold', 'ssp_num','markssp_num','before_Acc', 'before_Acc_class','before_mIoU','before_fwIoU','after_Acc', 'after_Acc_class','after_mIoU','after_fwIoU'])
     # unet模型不确定性度量阈值设置对应切片所有像素不确定性数据的百分位，可参考模型的像素精度，如像素精度为85%，阈值设置为15%
    # distarr=0.15
    log_file.write('LvSL percentile: %s '%distarr)
    for filename in file_list:
        print(filename)
        row = filename.split('_')[4]
        col = filename.split('_')[5]
        xoff = int(filename.split('_')[7])
        yoff = int(filename.split('_')[8].split('.')[0])
        gtfile = os.path.join(gt_dir, 'top_potsdam_%s_%s_label.tif'%(row,col))
        unet_predf=os.path.join(root_dir, filename.replace('softmax_prob','pred'))
        predds = gdal.Open(unet_predf)
        predarr = predds.ReadAsArray()
        gtds = gdal.Open(gtfile)
        gtarr = gtds.ReadAsArray(xoff, yoff, 512,512)
        log_file.write(filename)
    
        file = os.path.join(root_dir, filename)
        ds = gdal.Open(file)
        
        array = ds.ReadAsArray()
        distarray = distance_of_max_smax(array)
        # draw_histgram_of_array(distarray)
        mark,threhold = mark_uncertain_pixels(distarray, distarr)#distarr=0.2
        log_file.write(' LvSL threshold: %s '%threhold)
        
        segments_csv = os.path.join(segcsv,filename.replace('softmax_prob_','ergc').replace('.tif','.csv'))
        
        seg_df=pd.read_csv(segments_csv, sep=',',header=None)
        segments_mask=seg_df.values
        
        hcarr = mark
        hcsegments =segments_mask[hcarr==1] 
        hcsegids = set(list(hcsegments))
        print('the number of segments is ',len(hcsegids))
        log_file.write(' hc_superpixels number: %s '%len(hcsegids))
        ssp_num=len(hcsegids)
        list_hcsegids=list(hcsegids)
        
        #保留 superpixels 中超过80%的像素都是hard to classify的superpixel
        for hcsegid in list_hcsegids:
            hcpixel_num = hcsegments[hcsegments==hcsegid].sum()
            segpixel_num=segments_mask[segments_mask==hcsegid].sum()
            if hcpixel_num <=0.2*segpixel_num:
                hcsegids.remove(hcsegid)
        
        hc_seg=np.full(segments_mask.shape, 9999, dtype=int)
        # hc_seg_annotate = np.copy(predarr)
        hc_seg_annotate=np.full(segments_mask.shape, 9999, dtype=int)
        print('the number of segments needed to be labeled is ',len(hcsegids))
        markssp_num=len(hcsegids)
        
        log_file.write(' tomark_superpixels number: %s '%len(hcsegids))
        for hcsegid in list(hcsegids):
            # 标记hard to classify superpixels
            hc_seg[segments_mask==hcsegid]=hcsegid
            
            # 将非hard to classify superpixels 赋值为unet predict的类别值            
            seg_gtarr= gtarr[segments_mask==hcsegid]
            seg_gtlabel=most_frequent_in_array(seg_gtarr)
            
            seg_predarr= predarr[segments_mask==hcsegid]
            seg_predlabel=most_frequent_in_array(seg_predarr)

            hc_seg_annotate[segments_mask==hcsegid]=seg_gtlabel
        hc_seg_annotate[hc_seg_annotate==9999]=predarr[hc_seg_annotate==9999]
        # assert hc_seg_annotate.all() == predarr.all()
        # array_to_tiff(hc_seg, predds.GetProjection(), predds.GetGeoTransform(), 1,os.path.join(root_dir,filename.replace('softmax_prob','hcseg')))
        # array_to_tiff(hc_seg_annotate, predds.GetProjection(), predds.GetGeoTransform(), 1,os.path.join(root_dir,filename.replace('softmax_prob','hcseg_annote')))
        
        # 衡量人工标注unet hard to classify的superpixels的精度
        
        evaluator = Evaluator(6)
        evaluator.reset()
        evaluator.add_batch(gtarr, hc_seg_annotate)

        # Fast test during the training
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU =evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()  # 频权交并比
        print("after annotate: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        # print('the miou is %s'%myMIoU)
        log_file.write(" after annotate: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU)+'\n')
        
        evaluator1 = Evaluator(6)
        evaluator1.reset()
        evaluator1.add_batch(gtarr, predarr)
        # Fast test during the training
        Acc1 = evaluator1.Pixel_Accuracy()
        Acc_class1 = evaluator1.Pixel_Accuracy_Class()
        mIoU1 =evaluator1.Mean_Intersection_over_Union()
        FWIoU1 = evaluator1.Frequency_Weighted_Intersection_over_Union()  # 频权交并比
        print("before annotate: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc1, Acc_class1, mIoU1, FWIoU1))
        # print('the miou is %s'%myMIoU)
        new=pd.DataFrame({'filename':filename, 'threshold':threhold, 'ssp_num':ssp_num,'markssp_num':markssp_num,'before_Acc':Acc1, 'before_Acc_class':Acc_class1,'before_mIoU':mIoU1,'before_fwIoU':FWIoU1,'after_Acc':Acc, 'after_Acc_class':Acc_class,'after_mIoU':mIoU,'after_fwIoU':FWIoU},index = [0])
        records=records.append(new,ignore_index=True) 
        log_file.write("before annotate: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc1, Acc_class1, mIoU1, FWIoU1)+'\n')
    records.to_csv(os.path.join(os.getcwd(),'hclog_%s.csv'%str(distarr)))
    log_file.close()  
def polygonize_raster(inRasterfile, outShapefile):
    src_ds = gdal.Open(inRasterfile)
    if src_ds is None:
        print('Unable to open %s' % inRasterfile)
        sys.exit(1)
    srcband = src_ds.GetRasterBand(1)
    srs_wkt=src_ds.GetProjectionRef()
    
    driname = outShapefile.split('.')[-1]
    if driname=='shp':
        outDriver = ogr.GetDriverByName("ESRI Shapefile")
    elif driname =='geojson':
        outDriver = ogr.GetDriverByName("geojson")
    else:
        print('the vector format is not supported, only shp and geojson')
        sys.exit(1)
    if os.path.exists(outShapefile):
        outDriver.DeleteDataSource(outShapefile)
    outDataSource = outDriver.CreateDataSource(outShapefile)
    
    spatialRef = osr.SpatialReference()
    spatialRef.ImportFromWkt(srs_wkt)
    if spatialRef is None:
        print('no projection information')
        sys.exit(1)
    outLayer = outDataSource.CreateLayer('segs', srs=spatialRef)

    idField = ogr.FieldDefn('id', ogr.OFTInteger)
    outLayer.CreateField(idField)
    dst_field = outLayer.GetLayerDefn().GetFieldIndex('id')
    print(dst_field)
    gdal.Polygonize(srcband, None, outLayer, dst_field, [], callback=None)
    outDataSource.Destroy()

    srcband = None
    src_ds = None
    return
def polygonize_array(gt, proj, array, outShapefile):
    tmpRasterfile='E:/tmp/segraster.tif'
    xsize, ysize = array.shape
    dst_format = 'GTiff'
    dst_nbands = 1
    dst_datatype = gdal.GDT_Int16  # GDT_Float32
    
    driver = gdal.GetDriverByName(dst_format)
    dst_ds = driver.Create(tmpRasterfile, ysize, xsize, dst_nbands, dst_datatype)
    dst_ds.SetGeoTransform(gt)
    dst_ds.SetProjection(proj)
    dst_ds.GetRasterBand(1).WriteArray(array)
    dst_ds = None
#     cmd='gdal_polygonize.py %s -f GEOJSON  %s' % (dst_file,jsonfile)
#     os.system(cmd)
    polygonize_raster(tmpRasterfile, outShapefile)
    cmd = 'rm %s'%tmpRasterfile
    print(cmd)
    os.system(cmd)
    #             cmd='gdal_polygonize.py %s -f GEOJSON  %s' % (dst_file,jsonfile)
    #             os.system(cmd)
#     return vector_file 
    del dst_ds
    return

def bvsb_ssp_select():
    # root_dir='/home/zjh/tmp/unet_predict'
    # segcsv = '/home/zjh/Inria/AerialImageDataset/chicago_tiles/slic_markcsv'
    # gt_dir = '/home/zjh/Inria/AerialImageDataset/train/gt'
    root_dir = 'E:/tmp/unet_predict/'
    segcsv='E:/DATA/Inria/random_select_chicago/slic_markcsv/'
    gt_dir = 'E:/DATA/Inria/AerialImageDataset/train/gt/'
    # 'probdist_chicago1_0_3696.tif
    file_list = list(filter(lambda x: x.startswith('softmax_prob') and x.endswith('23_462_462.tif'), os.listdir(root_dir)))
       
     # log
    logfile = os.path.join(os.getcwd(),'hclog.txt')
    log_file = open(logfile, 'a')

     # unet模型不确定性度量阈值设置对应切片所有像素不确定性数据的百分位，如阈值设置为20%
    distarr=0.2
    log_file.write('BvSB percentile: %s '%distarr)
      
    for filename in file_list:
        print(filename)
        outShpfile=os.path.join('E:/tmp/hcsegshp',filename.replace('.tif','.geojson'))
        
        fstr=filename.split('.')[0]
        xoff = int(fstr.split('_')[3])
        yoff = int(fstr.split('_')[4])
        gtfile = os.path.join(gt_dir, fstr.split('_')[2]+'.tif')
        gtds = gdal.Open(gtfile)
        gtarr = gtds.ReadAsArray(xoff, yoff, 512,512)
        tmp = np.zeros(list(gtarr.shape))
        tmp[gtarr==255]=1
        gtarr=tmp
        
        unet_predf=os.path.join(root_dir, filename.replace('softmax_prob','pred'))
        predds = gdal.Open(unet_predf)
        predarr = predds.ReadAsArray()
          
        log_file.write(filename)
    
        file = os.path.join(root_dir, filename)
        ds = gdal.Open(file)
        array = ds.ReadAsArray()
        distarray = distance_of_max_smax(array)
        
        mark,threhold = mark_uncertain_pixels(distarray, distarr)#distarr=0.2
        log_file.write(' BvSB threshold: %s '%threhold)
        
        # chicago1_0_3696.csv
        segments_csv = os.path.join(segcsv,filename.replace('softmax_prob_','').replace('.tif','.csv')) 
        seg_df=pd.read_csv(segments_csv, sep=',',header=None)
        segments_mask=seg_df.values
        
        hcarr = mark
        # uncertain pixels
        hcsegments =segments_mask[hcarr==1] 
        hcsegids = set(list(hcsegments))
        ssp_num=len(hcsegids)
        log_file.write(' hc_superpixels number: %s '%ssp_num)
        list_hcsegids=list(hcsegids)
        
        #保留 superpixels 中超过20%的像素都是不确定像素
        for hcsegid in list_hcsegids:
            # number of uncertain pixels in a superpixel
            hcpixel_num = hcsegments[hcsegments==hcsegid].sum()
            # number of pixels in a superpixel
            segpixel_num=segments_mask[segments_mask==hcsegid].sum()
            if hcpixel_num <=0.2*segpixel_num:
                hcsegids.remove(hcsegid)
       
       # 保留superpixels中主要类别的面积占比大于80% 
        for hcsegid in list(hcsegids):
             #对superpixels中pred的值，统计各个元素出现次数
            seg_predarr= predarr[segments_mask==hcsegid]
            l = sorted([(np.sum(seg_predarr==i),i) for i in set(seg_predarr.flat)])
            # np.sum(b==i) #统计b中等于i的元素个数
            # set(b.flat)  #将b转为一维数组后，去除重复元素
            # sorted()     #按元素个数从小到大排序
            # l[-1]        #取出元素个数最多的元组对 (count,element)
            # seg_predlabel=most_frequent_in_array(seg_predarr)
            main_class_num, main_class_id = l[-1]
            
            if main_class_num<=0.8*len(seg_predarr):
                hcsegids.remove(hcsegid)
        print('the number of segments needed to be labeled is ',len(hcsegids))
        
        hc_seg_annotate=np.full(segments_mask.shape, 9999, dtype=int)
        hc_seg=np.full(segments_mask.shape, 9999, dtype=int)
        for hcsegid in list(hcsegids):
            hc_seg[segments_mask==hcsegid]=hcsegid
            # 将非hard to classify superpixels 赋值为unet predict的类别值            
            seg_gtarr= gtarr[segments_mask==hcsegid]
            seg_gtlabel=most_frequent_in_array(seg_gtarr)
            hc_seg_annotate[segments_mask==hcsegid]=seg_gtlabel
        hc_seg_annotate[hc_seg_annotate==9999]=predarr[hc_seg_annotate==9999]
        array_to_tiff(hc_seg_annotate, predds.GetProjection(), predds.GetGeoTransform(), 1,os.path.join(root_dir,filename.replace('softmax_prob','uncertseg_annote')))
        
        # 将待标注超像素hcseg到处为矢量多边形文件
        geotrans = predds.GetGeoTransform()
        gt = list(geotrans)
        proj = predds.GetProjection()
        tmpShpfile='E:/tmp/tmp.shp'
        polygonize_array(gt, proj, hc_seg, tmpShpfile)
        geojson_attr_process(tmpShpfile,outShpfile)
        
        # # hc_seg_annotate=np.full(segments_mask.shape, 9999, dtype=int)
        # # hc_seg_annotate = np.copy(predarr)
        # # hc_seg_annotate[hc_seg_annotate==9999]=predarr[hc_seg_annotate==9999]
        # # # assert hc_seg_annotate.all() == predarr.all()
        # # # 
        # # array_to_tiff(hc_seg_annotate, predds.GetProjection(), predds.GetGeoTransform(), 1,os.path.join(root_dir,filename.replace('softmax_prob','hcseg_annote')))
        
        
  
        # hc_seg=np.full(segments_mask.shape, 9999, dtype=int)
        # # hc_seg_annotate = np.copy(predarr)
        # hc_seg_annotate=np.full(segments_mask.shape, 9999, dtype=int)
        
        # markssp_num=len(hcsegids)
        
        # log_file.write(' tomark_superpixels number: %s '%len(hcsegids))
        # for hcsegid in list(hcsegids):
        #     # 标记hard to classify superpixels
        #     hc_seg[hcsegments==hcsegid]=hcsegid
        #     m = (hcarr==1) + (segments_mask==hcsegid)
        #     # 将非hard to classify superpixels 赋值为unet predict的类别值            
        #     seg_gtarr= gtarr[m]
        #     seg_gtlabel=most_frequent_in_array(seg_gtarr)
            
        #     seg_predarr= predarr[m]
        #     seg_predlabel=most_frequent_in_array(seg_predarr)

        #     hc_seg_annotate[m]=seg_gtlabel
        # hc_seg_annotate[hc_seg_annotate==9999]=predarr[hc_seg_annotate==9999]
        # # assert hc_seg_annotate.all() == predarr.all()
        # array_to_tiff(hc_seg, predds.GetProjection(), predds.GetGeoTransform(), 1,os.path.join(root_dir,filename.replace('softmax_prob','hcseg')))
        # array_to_tiff(hc_seg_annotate, predds.GetProjection(), predds.GetGeoTransform(), 1,os.path.join(root_dir,filename.replace('probdist','uncertainseg_annote')))
        
    #     # 衡量人工标注unet hard to classify的superpixels的精度
        
    #     evaluator = Evaluator(6)
    #     evaluator.reset()
    #     evaluator.add_batch(gtarr, hc_seg_annotate)

    #     # Fast test during the training
    #     Acc = evaluator.Pixel_Accuracy()
    #     Acc_class = evaluator.Pixel_Accuracy_Class()
    #     mIoU =evaluator.Mean_Intersection_over_Union()
    #     FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()  # 频权交并比
    #     print("after annotate: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
    #     # print('the miou is %s'%myMIoU)
    #     log_file.write(" after annotate: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU)+'\n')
        
    #     evaluator1 = Evaluator(6)
    #     evaluator1.reset()
    #     evaluator1.add_batch(gtarr, predarr)
    #     # Fast test during the training
    #     Acc1 = evaluator1.Pixel_Accuracy()
    #     Acc_class1 = evaluator1.Pixel_Accuracy_Class()
    #     mIoU1 =evaluator1.Mean_Intersection_over_Union()
    #     FWIoU1 = evaluator1.Frequency_Weighted_Intersection_over_Union()  # 频权交并比
    #     print("before annotate: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc1, Acc_class1, mIoU1, FWIoU1))
    #     # print('the miou is %s'%myMIoU)
    #     new=pd.DataFrame({'filename':filename, 'threshold':threhold, 'ssp_num':ssp_num,'markssp_num':markssp_num,'before_Acc':Acc1, 'before_Acc_class':Acc_class1,'before_mIoU':mIoU1,'before_fwIoU':FWIoU1,'after_Acc':Acc, 'after_Acc_class':Acc_class,'after_mIoU':mIoU,'after_fwIoU':FWIoU},index = [0])
    #     records=records.append(new,ignore_index=True) 
    #     log_file.write("before annotate: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc1, Acc_class1, mIoU1, FWIoU1)+'\n')
    # records.to_csv(os.path.join(os.getcwd(),'hc_ssp_select_%s.csv'%str(distarr)))
    #保存每个分割文件的不确定超像素id
    
    # import json  
    # print(hcsegid_dict)
    # # Data to be written  
    # with open("/home/zjh/Inria/AerialImageDataset/chicago_tiles/83epoches_20_30_hcsegids.json", "w") as outfile:  
    #     json.dump(hcsegid_dict, outfile) 
    log_file.close()        
def bvsb_no_ssp():
    root_dir='/home/zjh/tmp/unet_predict'
    segcsv = '/home/zjh/Inria/AerialImageDataset/chicago_tiles/slic_markcsv'
    gt_dir = '/home/zjh/Inria/AerialImageDataset/train/gt'
       
    # 'softmax_prob_top_potsdam_7_13_RGBIR_462_0'
    file_list = list(filter(lambda x: x.startswith('softmax_prob_chicago1_0_0') and x.endswith('.tif') , os.listdir(root_dir)))
     # log
    logfile = os.path.join(os.getcwd(),'hclog.txt')
    log_file = open(logfile, 'a')
    # records = pd.DataFrame(columns = ['filename', 'threshold', 'ssp_num','markssp_num','before_Acc', 'before_Acc_class','before_mIoU','before_fwIoU','after_Acc', 'after_Acc_class','after_mIoU','after_fwIoU'])
     # unet模型不确定性度量阈值设置对应切片所有像素不确定性数据的百分位，可参考模型的像素精度，如像素精度为85%，阈值设置为15%
    distarr=0.2
    log_file.write('BvSB percentile: %s '%distarr)
    for filename in file_list:
        print(filename)
        row = filename.split('_')[4]
        col = filename.split('_')[5]
        xoff = int(filename.split('_')[7])
        yoff = int(filename.split('_')[8].split('.')[0])
        gtfile = os.path.join(gt_dir, 'top_potsdam_%s_%s_label.tif'%(row,col))
        unet_predf=os.path.join(root_dir, filename.replace('softmax_prob','pred'))
        predds = gdal.Open(unet_predf)
        predarr = predds.ReadAsArray()
        gtds = gdal.Open(gtfile)
        gtarr = gtds.ReadAsArray(xoff, yoff, 512,512)
        
        log_file.write(filename)
    
        file = os.path.join(root_dir, filename)
        ds = gdal.Open(file)
        
        array = ds.ReadAsArray()
        distarray = distance_of_max_smax(array)
        # draw_histgram_of_array(distarray)
        mark,threhold = mark_uncertain_pixels(distarray, distarr)#distarr=0.2
        log_file.write(' LvSL threshold: %s '%threhold)
        
        segments_csv = os.path.join(segcsv,filename.replace('softmax_prob_','ergc').replace('.tif','.csv'))
        
        seg_df=pd.read_csv(segments_csv, sep=',',header=None)
        segments_mask=seg_df.values
        
        hcarr = mark
        hcsegments =segments_mask[hcarr==1] 
        hcsegids = set(list(hcsegments))
        print('the number of segments is ',len(hcsegids))
        log_file.write(' hc_superpixels number: %s '%len(hcsegids))
        ssp_num=len(hcsegids)
        list_hcsegids=list(hcsegids)
        
        # #保留 superpixels 中超过80%的像素都是hard to classify的superpixel
        # for hcsegid in list_hcsegids:
        #     hcpixel_num = hcsegments[hcsegments==hcsegid].sum()
        #     segpixel_num=segments_mask[segments_mask==hcsegid].sum()
        #     if hcpixel_num <=0.2*segpixel_num:
        #         hcsegids.remove(hcsegid)
        
        hc_seg=np.full(segments_mask.shape, 9999, dtype=int)
        # hc_seg_annotate = np.copy(predarr)
        hc_seg_annotate=np.full(segments_mask.shape, 9999, dtype=int)
        print('the number of segments needed to be labeled is ',len(hcsegids))
        markssp_num=len(hcsegids)
        
        log_file.write(' tomark_superpixels number: %s '%len(hcsegids))
        for hcsegid in list(hcsegids):
            # 标记hard to classify superpixels
            # hc_seg[hcsegments==hcsegid]=hcsegid
            m = (hcarr==1) + (segments_mask==hcsegid)
            # 将非hard to classify superpixels 赋值为unet predict的类别值            
            seg_gtarr= gtarr[m]
            seg_gtlabel=most_frequent_in_array(seg_gtarr)
            
            seg_predarr= predarr[m]
            seg_predlabel=most_frequent_in_array(seg_predarr)

            hc_seg_annotate[m]=seg_gtlabel
        hc_seg_annotate[hc_seg_annotate==9999]=predarr[hc_seg_annotate==9999]
        # assert hc_seg_annotate.all() == predarr.all()
        # array_to_tiff(hc_seg, predds.GetProjection(), predds.GetGeoTransform(), 1,os.path.join(root_dir,filename.replace('softmax_prob','hcseg')))
        array_to_tiff(hc_seg_annotate, predds.GetProjection(), predds.GetGeoTransform(), 1,os.path.join(root_dir,filename.replace('softmax_prob','hcseg_annote')))
        
        # 衡量人工标注unet hard to classify的superpixels的精度
        
        evaluator = Evaluator(6)
        evaluator.reset()
        evaluator.add_batch(gtarr, hc_seg_annotate)

        # Fast test during the training
        Acc = evaluator.Pixel_Accuracy()
        Acc_class = evaluator.Pixel_Accuracy_Class()
        mIoU =evaluator.Mean_Intersection_over_Union()
        FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()  # 频权交并比
        print("after annotate: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        # print('the miou is %s'%myMIoU)
        log_file.write(" after annotate: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU)+'\n')
        
        evaluator1 = Evaluator(6)
        evaluator1.reset()
        evaluator1.add_batch(gtarr, predarr)
        # Fast test during the training
        Acc1 = evaluator1.Pixel_Accuracy()
        Acc_class1 = evaluator1.Pixel_Accuracy_Class()
        mIoU1 =evaluator1.Mean_Intersection_over_Union()
        FWIoU1 = evaluator1.Frequency_Weighted_Intersection_over_Union()  # 频权交并比
        print("before annotate: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc1, Acc_class1, mIoU1, FWIoU1))
        # print('the miou is %s'%myMIoU)
        new=pd.DataFrame({'filename':filename, 'threshold':threhold, 'ssp_num':ssp_num,'markssp_num':markssp_num,'before_Acc':Acc1, 'before_Acc_class':Acc_class1,'before_mIoU':mIoU1,'before_fwIoU':FWIoU1,'after_Acc':Acc, 'after_Acc_class':Acc_class,'after_mIoU':mIoU,'after_fwIoU':FWIoU},index = [0])
        records=records.append(new,ignore_index=True) 
        log_file.write("before annotate: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc1, Acc_class1, mIoU1, FWIoU1)+'\n')
    records.to_csv(os.path.join(os.getcwd(),'hc_uncertian_ssp_%slog.csv'%str(distarr)))
    log_file.close() 
def random_select_ssp():
    root_dir = 'E:/tmp/unet_predict/'
    segcsv='E:/DATA/Inria/random_select_chicago/slic_markcsv/'
    gt_dir = 'E:/DATA/Inria/AerialImageDataset/train/gt/'
    hcsegshp='E:/tmp/hcsegshp'
    
    # 'probdist_chicago1_0_3696.tif
    file_list = list(filter(lambda x: x.startswith('softmax_prob') and x.endswith('.tif'), os.listdir(root_dir)))
       
     # log
    logfile = os.path.join(os.getcwd(),'hclog.txt')
    log_file = open(logfile, 'a')
    
    records = pd.DataFrame(columns = ['filename', 'markssp_num'])
    x=0
    n=len(file_list)
    for filename in file_list:
        print(filename)
        hcsegShpfile=os.path.join('E:/tmp/hcsegshp',filename.replace('.tif','.geojson'))
         # 计算该文件应该标注的超像素的个数
        gdf = geopandas.read_file(hcsegShpfile)
        num = len(gdf)
        x+=num
    print('mean number of labeling pixels is',x/n)
    '''
        log_file.write('random select ssp: %s '%num)
        fstr=filename.split('.')[0]
        xoff = int(fstr.split('_')[3])
        yoff = int(fstr.split('_')[4])
        gtfile = os.path.join(gt_dir, fstr.split('_')[2]+'.tif')
        gtds = gdal.Open(gtfile)
        gtarr = gtds.ReadAsArray(xoff, yoff, 512,512)
        tmp = np.zeros(list(gtarr.shape))
        tmp[gtarr==255]=1
        gtarr=tmp
        
        unet_predf=os.path.join(root_dir, filename.replace('softmax_prob','pred'))
        predds = gdal.Open(unet_predf)
        predarr = predds.ReadAsArray()
        
        log_file.write(filename)
        
        segments_csv = os.path.join(segcsv,filename.replace('softmax_prob_','').replace('.tif','.csv')) 
        seg_df=pd.read_csv(segments_csv, sep=',',header=None)
        segments_mask=seg_df.values
        
        segids = [i for i in set(segments_mask.flat)]
        random_select_segids= random.sample(segids,num)
        
        seg_annotate=np.full(segments_mask.shape, 9999, dtype=int)
        for segid in random_select_segids:
            # 将非hard to classify superpixels 赋值为unet predict的类别值            
            seg_gtarr= gtarr[segments_mask==segid]
            l = sorted([(np.sum(seg_gtarr==i),i) for i in set(seg_gtarr.flat)])
            # np.sum(b==i) #统计b中等于i的元素个数
            # set(b.flat)  #将b转为一维数组后，去除重复元素
            # sorted()     #按元素个数从小到大排序
            # l[-1]        #取出元素个数最多的元组对 (count,element)
            # seg_predlabel=most_frequent_in_array(seg_predarr)
            main_class_num, seg_gtlabel = l[-1]
            # seg_gtlabel=most_frequent_in_array(seg_gtarr)

            seg_annotate[segments_mask==segid]=seg_gtlabel
        seg_annotate[seg_annotate==9999]=predarr[seg_annotate==9999]
        array_to_tiff(seg_annotate, predds.GetProjection(), predds.GetGeoTransform(), 1,os.path.join(root_dir,filename.replace('softmax_prob','randomseg_annote')))
                
                
        # evaluator = Evaluator(6)
        # evaluator.reset()
        # evaluator.add_batch(gtarr, seg_annotate)

        # # Fast test during the training
        # Acc = evaluator.Pixel_Accuracy()
        # Acc_class = evaluator.Pixel_Accuracy_Class()
        # mIoU =evaluator.Mean_Intersection_over_Union()
        # FWIoU = evaluator.Frequency_Weighted_Intersection_over_Union()  # 频权交并比
        # print("after annotate: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU))
        # # print('the miou is %s'%myMIoU)
        # log_file.write(" after annotate: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc, Acc_class, mIoU, FWIoU)+'\n')
        
        # evaluator1 = Evaluator(6)
        # evaluator1.reset()
        # evaluator1.add_batch(gtarr, predarr)
        # # Fast test during the training
        # Acc1 = evaluator1.Pixel_Accuracy()
        # Acc_class1 = evaluator1.Pixel_Accuracy_Class()
        # mIoU1 =evaluator1.Mean_Intersection_over_Union()
        # FWIoU1 = evaluator1.Frequency_Weighted_Intersection_over_Union()  # 频权交并比
        # print("before annotate: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc1, Acc_class1, mIoU1, FWIoU1))
        # print('the miou is %s'%myMIoU)
        new=pd.DataFrame({'filename':filename, 'markssp_num':num},index = [0])
        records=records.append(new,ignore_index=True) 
        # log_file.write("before annotate: Acc:{}, Acc_class:{}, mIoU:{}, fwIoU: {}".format(Acc1, Acc_class1, mIoU1, FWIoU1)+'\n')
    records.to_csv(os.path.join(os.getcwd(),'random_select_%s.csv'%str(num)))
    # log_file.close()
    '''
def unet_predict(array,model):
#     gc.enable()
#     gc.set_debug(gc.DEBUG_UNCOLLECTABLE ) # gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_STATS | 
    
    # Define network
    # net = UNet_SNws(3, 64, 6, using_movavg=1, using_bn=1)
    
    # model = os.path.join(os.getcwd(),'run/lulc/unetv2/model_best.pth.tar')
    net = UNet_SNws(3, 64, 2, using_movavg=1, using_bn=1)
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

def cal_class_prob():
    import pandas as pd
    import cv2
    import json
    tile_size=512
    
    random_chicago_tiles = '/home/zjh/Inria/AerialImageDataset/chicago_tiles/tif'
    tile_files=os.listdir(random_chicago_tiles)
    model = '/home/zjh/phd_annotate/run/inria/unet/178epoch_model_best.pth.tar'
    
    data_path = '/home/zjh/Inria/AerialImageDataset/train/images'
    label_path = '/home/zjh/Inria/AerialImageDataset/train/gt'
    
    result_path='/home/zjh/tmp/unet_predict'
    score=0
    a_score=0
    i=0
    distarr=0.2
    for file in tile_files:
        i=i+1
        ds = gdal.Open(os.path.join(random_chicago_tiles,file))
        img_tif = ds.ReadAsArray()

        outputs = unet_predict(img_tif,model)
        
        outputarr = torch.squeeze(outputs, 0).detach().cpu().numpy()
        # print(outputarr.shape)
        # print(outputarr[:,10,10])

        _, predict = torch.max(outputs, 1)
        pred_ = predict[0].numpy()
        distarray = distance_of_max_smax(outputarr)
        
        mark,threhold = mark_uncertain_pixels(distarray, distarr)#distarr=0.2
        uncertain_pixels = mark.astype(int)
        filenamestr = file.split('.')[0]
        xoff = int(filenamestr.split('_')[1])
        yoff = int(filenamestr.split('_')[2])
        out_proj=ds.GetProjection()
        gt = ds.GetGeoTransform()
        # array_to_tiff(outputarr,out_proj, gt, outputarr.shape[0], os.path.join(result_path,'softmax_prob_'+file))
    #     # print(pred.max())
    #     print('[%s] Finished test.' % datetime.now())
        # array_to_tiff(uncertain_pixels, out_proj, gt, 1, os.path.join(result_path,'uncertain_pixels_'+file))
        array_to_tiff(pred_, out_proj, gt, 1, os.path.join(result_path,'pred_'+file))
        

if __name__ == '__main__':
    # main(0.25)#超像素筛选+超像素扩展，标注所有包含80%以上不确定像素的超像素,模型总体精度为0.85，这里设为0.15
    # main(0.05)
    # cal_class_prob()
    
    # bvsb_ssp_select()#超像素筛选，标注所有包含20%以上不确定像素的超像素的不确定像素
    random_select_ssp()
    # bvsb_no_ssp()#以超像素为单位，标注所有包含不确定区域的超像素的不确定像素
    
    # mark_hard_to_classify_by_probarray('/home/zjh/tmp/unet_predict', 0.2))
    # segments_csv = 'E:/tmp/ergc_test/maskcsv/ergctop_potsdam_2_13_RGBIR_3234_1386.csv'
    # hard_to_classify_f = 'E:/tmp/ergc_test/unet_predict/maxd_top_potsdam_2_13_RGBIR_3234_1386.tif'
    # find_hard_to_classify_segments(segments_csv,hard_to_classify_f)