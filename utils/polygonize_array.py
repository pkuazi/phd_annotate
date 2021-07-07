import gdal, ogr, osr
import sys, os

def create_shpfile(dstfolder, input_wkt):  
    geom = ogr.CreateGeometryFromWkt(input_wkt)
    
    outShapefile = os.path.join(dstfolder, "clip_extent.shp")
    outDriver = ogr.GetDriverByName("ESRI Shapefile")
    
    if os.path.exists(outShapefile):
        outDriver.DeleteDataSource(outShapefile)
    
    outDataSource = outDriver.CreateDataSource(outShapefile)
    
    spatialRef = osr.SpatialReference()
    spatialRef.ImportFromEPSG(4326)
    outLayer = outDataSource.CreateLayer("clip_extent", geom_type=ogr.wkbPolygon, srs = spatialRef)
    
    idField = ogr.FieldDefn("id", ogr.OFTInteger)
    outLayer.CreateField(idField)
    
    featureDefn = outLayer.GetLayerDefn()
    feature = ogr.Feature(featureDefn)
    feature.SetGeometry(geom)
    feature.SetField("id", 1)
    outLayer.CreateFeature(feature)
    
    outDataSource.Destroy()
    
    return outShapefile

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
    outLayer = outDataSource.CreateLayer('ergcsegs', srs=spatialRef)

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

def main(inMaskcsvfile,refimagefile, outShpfile):
    import pandas as pd
#     df = pd.read_csv('/tmp/ssp/ergc_rgb_400_csv/ergc_rgb_400Jingjinji_7_15_512_11776.csv', sep=',', header=None)
    df = pd.read_csv(inMaskcsvfile, sep=',', header=None)
    ergc_array = df.values
    
    ds_img = gdal.Open(refimagefile) 
    xsize = ds_img.RasterXSize
    ysize = ds_img.RasterYSize
    geotrans = ds_img.GetGeoTransform()
    gt = list(geotrans)
    proj = ds_img.GetProjection()
    
    # xoff = 11776
    # yoff = 6656
    # gt[0] = geotrans[0] + xoff * geotrans[1]
    # gt[3] = geotrans[3] + yoff * geotrans[5]
   
    polygonize_array(gt, proj, ergc_array, outShpfile)
if __name__ == '__main__':
    mask_dir = 'E:/tmp/ergc_test/maskcsv'
    files = list(filter(lambda x: x.endswith('.csv'), os.listdir(mask_dir)))
    import json
    with open('/home/zjh/Inria/AerialImageDataset/chicago_tiles/70epoches_20_30_hcsegids.json', "r") as f:    #打开文件
        data = f.read()   #读
        hcsegids_dict = json.loads(data)
    
    for csvfile in files:
        hcsegids = hcsegids_dict[csvfile]
        mask_csvfile=os.path.join(mask_dir,csvfile)
        refimagefile = os.path.join('E:/tmp/ergc_test/tif',csvfile.replace('csv','tif')[4:])
        out_shpfile = os.path.join('E:/tmp/ergc_test/segpolys',csvfile.replace('csv','shp')[4:])
        main(mask_csvfile,hcsegids, refimagefile,out_shpfile)
    
    # import cv2
    # bands_data = cv2.imread('/tmp/jjj_2m/Jingjinji_7_15_12288_6144.jpg')
#     from image_segmentation import segments_classify
#     segments_class = segments_classify(bands_data, ergc_array,6)
