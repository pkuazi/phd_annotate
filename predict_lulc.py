import os
import numpy as np
import torch
from modeling.deeplab import *
import gc 
from modeling.UNet_SNws import *
import time
from datetime import datetime 
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# Test the trained model
def unet_predict(array):
#     gc.enable()
#     gc.set_debug(gc.DEBUG_UNCOLLECTABLE ) # gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_STATS | 
    
    # Define network
    # net = UNet_SNws(3, 64, 6, using_movavg=1, using_bn=1)
    
    # model = os.path.join(os.getcwd(),'run/lulc/unetv2/model_best.pth.tar')
    net = UNet_SNws(4, 64, 2, using_movavg=1, using_bn=1)
    model = os.path.join(os.getcwd(), 'run/potsdam/unet/model_best.pth.tar')
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
    del inputs
    print(outputs.shape)
    score, predict = torch.max(outputs, 1)
    del outputs
        
    pred = predict[0].numpy()
    print(pred.max())
    print('[%s] Finished test.' % datetime.now())
    
    del net 
    return pred
def deeplabv3_predict(array):
#     gc.enable()
#     gc.set_debug(gc.DEBUG_UNCOLLECTABLE ) # gc.DEBUG_COLLECTABLE | gc.DEBUG_UNCOLLECTABLE | gc.DEBUG_STATS | 
    
    # Define network
    net = DeepLab(num_classes=6,
                    backbone='xception',
                    output_stride=16,
                    sync_bn=None,
                    freeze_bn=False)
    
    model = os.path.join(os.getcwd(),'run/lulc/deeplabv3+xception/experiment_14/checkpoint.pth.tar')
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
    del inputs
    
    _, predict = torch.max(outputs.data, 1)
    del outputs
        
    pred = predict[0].numpy()
    print('[%s] Finished test.' % datetime.now())
    
    del net 
    return pred
    
if __name__ == '__main__':  
    input_path = '/home/zjh/ISPRS_BENCHMARK_DATASETS/4_Ortho_RGBIR/4_Ortho_RGBIR/top_potsdam_2_10_RGBIR.tif'
    import gdal,cv2
    ds = gdal.Open(input_path)
    # img = ds.ReadAsArray(50,60,256,256)
    img_tif = ds.ReadAsArray(50,60,512,512)
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
    pred_arr = unet_predict(img_tif)
#     pred_arr = deeplabv3_predict(img)
    from PIL import Image
    pred_arr = pred_arr.astype('int8')
    im = Image.fromarray(pred_arr)
    im =im.convert("L")
    im.save('/home/zjh/tmp/unetv2_pts2_10.jpg')

#     array1 = ds.ReadAsArray(500,200, 256,256)
#     array1 = np.array(array1, dtype='int16')
#     pred_a = deeplabv3_predict(array1)
    
  
    # input_path = '/home/zjh/LULC_CNIC_BAIDU/img_small/'
    # for i in range(10):
    #     img = os.path.join(input_path, 'T01000%s.jpg'%i)
    #     ds = gdal.Open(img)
    #     img = ds.ReadAsArray()
    #     img = cv2.normalize(img,img,alpha=0, beta=255, norm_type=cv2.NORM_MINMAX,dtype=cv2.CV_32F)
        
    # #     img = '/tmp/data/6405_2015_56_02_LC81290342015244LGN00.jpg'
    # #     import cv2 
    # #     array = cv2.imread(img,-1)
    # #     array = np.transpose(array,(2,0,1))
    #     # pred_arr = unet_predict(img)
    #     pred_arr = deeplabv3_predict(img)
    #     from PIL import Image
    #     pred_arr = pred_arr.astype('int8')
    #     im = Image.fromarray(pred_arr)
    #     im =im.convert("L")
    #     im.save(os.path.join('/home/zjh/tmp/T01000%s.jpg'%i))
    # del ds

#     
#     gc.collect()
#     time.sleep(5)
    



    
    
