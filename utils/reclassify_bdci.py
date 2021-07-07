import os
import cv2
import gdal
from PIL import Image
import numpy as np

input='/home/zjh/CCF-BDCI/lab_train'
output = '/home/zjh/LULC_CNIC_BAIDU/gt'

# 0 建筑
# 1 耕地
# 2 林地
# 3 水体
# 4 道路
# 5 草地
# 6 其他
# 255 未标注区域
in_classid = [0,1,2,3,4,5,6,255]

# 1 耕地
# 2 林地
# 3 草地
# 4 建筑
# 5 水体
# 6 其他
out_classid=[4,1,2,5,4,3,6,255]

files = os.listdir(input)
files = list(filter(lambda x: x.endswith('.png'), files))
for file in files:
    ds = gdal.Open(os.path.join(input, file))
    x = ds.ReadAsArray()
    tmp = np.zeros([256,256])
    tmp[x==0]=4
    tmp[x==1]=1
    tmp[x==2]=2
    tmp[x==3]=5
    tmp[x==4]=4
    tmp[x==5]=3
    tmp[x==6]=6
    tmp[x==255]=6
      
    im = Image.fromarray(tmp)
    im = im.convert('P')
    im.save(os.path.join(output,file))

