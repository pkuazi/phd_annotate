# -*- coding: utf-8 -*-
"""
Created on Sat Mar  6 03:45:19 2021

@author: ink
"""
#对根据原始切片影像生成的推荐标注的超像素边界多边形进行处理，转换投影为4326，并删除9999的字段。
import geopandas
from shapely import geometry
import os
def reprojection(invector, outvector):
    cmd = 'ogr2ogr %s -t_srs "EPSG:4326" %s '%(outvector, invector)
    os.system(cmd)
    
def geojson_attr_process(file_path,outvector):
    gdf = geopandas.read_file(file_path)
    # print(gdf.head())
    
    drop_list=[]
    for i in range(0, len(gdf) ):
        geo = gdf.geometry[i] #获取空间属性，即GeoSeries
        name = gdf.id[i]	
        if name==9999:
            drop_list.append(i)
    
    new_gdf = gdf.drop(drop_list)
            
    
    # # 投影转换
    # current_crs = gdf.crs
    # print(current_crs)
    # new_gdf.crs = current_crs
      
    #save
    new_gdf.to_file("E:/tmp/test.geojson", driver='GeoJSON')
    # new_gdf.to_file("E:/tmp/test.shp")
    
    reprojection("E:/tmp/test.geojson", outvector)

def saveShapefile(file_path, output_shapefile):
    '''
    
    Parameters
    ----------
    file_path : TYPE
       输入GeoJSON的名称，
    output_shapefile : TYPE
        输出shapfile的名称（默认投影为wgs1984）.

    Returns
    -------
    None.

    '''
    data = geopandas.read_file(file_path)
    data.to_file(output_shapefile, driver='ESRI Shapefile', encoding='utf-8')
    print("--保存成功，文件存放位置："+output_shapefile)


if __name__ == '__main__':
    infolder = 'E:/DATA/Inria/random_select_chicago/hcsegshp'
    outfolder = 'E:/DATA/Inria/random_select_chicago/processed_hcseggeojson'
    files = list(filter(lambda x:x.endswith('.geojson'), os.listdir(infolder)))
    for file in files:
        geojson_file = os.path.join(infolder,file)
        outputfile =os.path.join(outfolder, file) 
        geojson_attr_process(geojson_file, outputfile)
    # saveShapefile(geojson_file,'E:/tmp/uncertain_seg_chicago1_1386_462.shp')