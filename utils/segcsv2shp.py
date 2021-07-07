# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 05:21:15 2021

@author: ink
"""
import gdal,ogr,osr
import os,sys
import pandas as pd
from raster_io import array_to_tiff
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
if __name__ == '__main__':
    segcsv = '/home/zjh/tmp/maskcsv'
    output='E:/tmp'
    slic_seg_csv = 'E:/DATA/Inria/random_select_chicago/slic_markcsv/chicago1_1386_462.csv'
    seeds_seg_csv = 'E:/DATA/Inria/random_select_chicago/slic_600_markcsv/chicago1_1386_462.csv'
    imgds = gdal.Open('E:/DATA/Inria/random_select_chicago/tif/chicago1_1386_462.tif')
    seg_df=pd.read_csv(seeds_seg_csv, sep=',',header=None)
    segments_mask=seg_df.values
    output_segtif = os.path.join(output,'slic600seg_chicago1_1386_462.tif')
    array_to_tiff(segments_mask, imgds.GetProjection(), imgds.GetGeoTransform(), 1,output_segtif)
    output_segshp = os.path.join(output,'slic600seg_chicago1_1386_462.shp')
    polygonize_raster(output_segtif,output_segshp)