# from skimage.color.colorlabel import label2rgb

import gdal
import os
import numpy as np
import cv2
from PIL import Image

classid2rgb_map = {
    0: [255, 255, 255],  # "white", 地面
    1: [255, 0, 0],  # "red",房屋
    2: [0, 0, 255],  # "blue", 其他
    3: [0, 255, 255],  # "cyan", 车
    4: [0, 255, 0],  # "green", 树木
    5: [255, 255, 0],  # "yello" 草地
}
NP2GDAL_CONVERSION = {
  "uint8": 1,
  "int8": 1,
  "uint16": 2,
  "int16": 3,
  "uint32": 4,
  "int32": 5,
  "float32": 6,
  "float64": 7,
  "complex64": 10,
  "complex128": 11,
}
def label2rgb(pred_y):
    #print(set(list(pred_y.reshape(-1))))
    rgb_img = np.zeros((pred_y.shape[0], pred_y.shape[1], 3))
    for i in range(len(pred_y)):
        for j in range(len(pred_y[0])):
            rgb_img[i][j] = classid2rgb_map.get(pred_y[i][j], [255, 255, 255])
    return rgb_img.astype(np.uint8)

# tiff.imsave(dump_file_name, img)
def main():
    tifpath='/home/zjh/AIDataset/gt'
    tif = os.listdir(tifpath)
    # change grey to rgb
    # for file in tif:
    #     if file.endswith('.tif'):
    #         ds = gdal.Open(os.path.join(tifpath, file))
    #         data = ds.ReadAsArray()
    #         img = label2rgb(data)
    #         png_name = '/home/zjh/AIDataset/gt_png/%s.png'%file[:-4]
    #         #tiff.imsave(dump_file_name, img)
    #         print(png_name)
    #         cv2.imwrite(png_name, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
    
    input='/home/zjh/AIDataset/gt'
    output = '/home/zjh/LULC_CNIC_BAIDU/gt'
    files = os.listdir(input)
    files = list(filter(lambda x: x.endswith('.tif'), files))
    for file in files:
    # change tif to png
        ds = gdal.Open(os.path.join(input, file))
    
        data = ds.ReadAsArray()      
        im = Image.fromarray(data)
        im = im.convert('P')
        im.save(os.path.join(output,file.split('.')[0]+'.png'))
if __name__ == '__main__':   
    import json
    test_tiles_list = '/home/zjh/phd_annotate/run/test_file_list.txt'
    files_root = '/home/zjh/Inria/AerialImageDataset/train/images'
    b = open(test_tiles_list, "r",encoding='UTF-8')
    out_test = b.read()
    test_file_names = json.loads(out_test) 
    output='/home/zjh/Inria/AerialImageDataset/chicago_tiles'
    for tile in test_file_names:
        filename,xoff, yoff=tile
        imagefile=os.path.join(files_root,filename)
        # ds_img = gdal.Open(imagefile) 
        # data = ds_img.ReadAsArray(xoff,yoff, 512, 512)      
        # im = Image.fromarray(data)
        # im = im.convert('P')
        # im.save(os.path.join(output,file.split('.')[0]+'.png'))
        
#     imagefile='/mnt/win/data/Potsdam/4_Ortho_RGBIR/top_potsdam_4_13_RGBIR.tif'
#     labelfile = '/mnt/win/data/Potsdam/9_labels_id/top_potsdam_4_13_label_proj.tif'
# #     '"top_potsdam_4_13_RGBIR.tif", 2560, 5120]'
#     xoff = 2560
#     yoff = 5120 
        xsize=512
        ysize=512
    #     imagefile='/mnt/win/data/xiongan/Jingjinji_7_15_utm.tif'
    #     labelfile = '/mnt/win/data/xiongan/N50_35_2010_jjj_7_15_resample.tif'
    #     baifile = '/mnt/win/data/xiongan/BAI/LC08_L1TP_123033_20180408_20180417_01_T1_BAI.TIF'
        ds_img = gdal.Open(imagefile) 
        out_proj=ds_img.GetProjection()
        gt = ds_img.GetGeoTransform()
        out_gt = list(gt)
        out_gt[0] = gt[0] + xoff * gt[1]
        out_gt[3] = gt[3] + yoff * gt[5]
#         ds_label = gdal.Open(labelfile)
        data = ds_img.ReadAsArray(xoff,yoff, 512, 512)
        dst_nbands=data.shape[0]
        dst_format = 'GTiff'
        driver = gdal.GetDriverByName(dst_format)
        dtype = NP2GDAL_CONVERSION[str(data.dtype)]
        dst_ds = driver.Create(os.path.join(output,'%s_%s_%s.tif'%(filename.split('.')[0],xoff, yoff)), ysize, xsize, dst_nbands, dtype)
        dst_ds.SetGeoTransform(out_gt)
        dst_ds.SetProjection(out_proj)
    
        if dst_nbands == 1:
            dst_ds.GetRasterBand(1).WriteArray(data)
        else:
            for i in range(dst_nbands):
                dst_ds.GetRasterBand(i + 1).WriteArray(data[i, :, :])
        del dst_ds
    
#         label = ds_label.ReadAsArray(xoff,yoff, 512, 512)
        # band1=data[0]
        # band2=data[1]
        # band3=data[2]
            
        # imgcolor = np.dstack([band1,band2,band3])
        # cv2.imwrite('E:/tmp/test/png/%s_%s_%s.png'%(filename.split('.')[0],xoff, yoff),imgcolor)
