#
# -*-coding:utf-8 -*-
#
# @Author: zhaojianghua
# @Date  : 2018-12-19 09:32
#

"""

""" 
import os
from osgeo import gdal,osr
from shapely.geometry import mapping, Polygon
import fiona
import json

def GetExtent(gt,cols,rows):
    ''' Return list of corner coordinates from a geotransform

        @type gt:   C{tuple/list}
        @param gt: geotransform
        @type cols:   C{int}
        @param cols: number of columns in the dataset
        @type rows:   C{int}
        @param rows: number of rows in the dataset
        @rtype:    C{[float,...,float]}
        @return:   coordinates of each corner
    '''
    ext=[]
    xarr=[0,cols]
    yarr=[0,rows]

    for px in xarr:
        for py in yarr:
            x=gt[0]+(px*gt[1])+(py*gt[2])
            y=gt[3]+(px*gt[4])+(py*gt[5])
            ext.append([x,y])
            print(x,y)
        yarr.reverse()
    return ext

def ReprojectCoords(coords,src_srs,tgt_srs):
    ''' Reproject a list of x,y coordinates.

        @type geom:     C{tuple/list}
        @param geom:    List of [[x,y],...[x,y]] coordinates
        @type src_srs:  C{osr.SpatialReference}
        @param src_srs: OSR SpatialReference object
        @type tgt_srs:  C{osr.SpatialReference}
        @param tgt_srs: OSR SpatialReference object
        @rtype:         C{tuple/list}
        @return:        List of transformed [[x,y],...[x,y]] coordinates
    '''
    trans_coords=[]
    transform = osr.CoordinateTransformation( src_srs, tgt_srs)
    for x,y in coords:
        x,y,z = transform.TransformPoint(x,y)
        trans_coords.append([x,y])
    return trans_coords

def geojson2shp(geojson, shpdst, id):
    '''

    :param geojson: the geojson format of a polygon
    :param shpdst: the path of the shapefile
    :param id: the id property
    :return: no return, just save the shapefile into the shpdst
    '''
    # an example Shapely geometry
    coordinates = geojson['coordinates']
    poly = Polygon(coordinates[0])

    # Define a polygon feature geometry with one attribute
    schema = {
        'geometry': 'Polygon',
        'properties': {'id': 'int'},
    }

    # Write a new Shapefile
    with fiona.open(shpdst, 'w', 'ESRI Shapefile', schema) as c:
        ## If there are multiple geometries, put the "for" loop here
        c.write({
            'geometry': mapping(poly),
            'properties': {'id': id},
        })
def gen_bound_for_Potsdam():
    tif_folder = os.path.join(os.getcwd(), 'data/img')
    bound_folder = os.path.join(os.getcwd(), 'data/bound')
    if not os.path.exists(bound_folder):
        os.makedirs(bound_folder)
    
    file_names = os.listdir(tif_folder)
    file_names = list(filter(lambda x: x.endswith(".tif"), file_names))
    
    for file in file_names:
        file_path = os.path.join(tif_folder, file)
        file_name = file[:-7]
        bound_name = file_name+'bound.shp'
        bound_path = os.path.join(bound_folder, bound_name)
    
        ds = gdal.Open(file_path)
        gt = ds.GetGeoTransform()
        cols = ds.RasterXSize
        rows = ds.RasterYSize
    
        ext = GetExtent(gt, cols, rows)
    
        src_srs = osr.SpatialReference()
        src_srs.ImportFromWkt(ds.GetProjection())
        # tgt_srs=osr.SpatialReference()
        # tgt_srs.ImportFromEPSG(4326)
        tgt_srs = src_srs.CloneGeogCS()
        print(tgt_srs)
    
        crs = tgt_srs.ExportToProj4()
    
        geo_ext = ReprojectCoords(ext, src_srs, tgt_srs)
        print(geo_ext)
        poly = Polygon(geo_ext)
    
        # Define a polygon feature geometry with one attribute
        schema = {
            'geometry': 'Polygon',
            'properties': {'id': 'int'},
        }
    
        # Write a new Shapefile
        with fiona.open(bound_path, 'w', driver = 'ESRI Shapefile', schema=schema, crs=crs) as c:
            ## If there are multiple geometries, put the "for" loop here
            c.write({
                'geometry': mapping(poly),
                'properties': {'id': 1},
            })
if __name__ == "__main__":
    img_path = '/home/zjh/Inria/AerialImageDataset/train/images'
    dbox_path = '/home/zjh/Inria/AerialImageDataset/train/dbox'
    tiles_list='/home/zjh/test_file_list.txt'
    
    bound_file = '/home/zjh/Inria/AerialImageDataset/train/tile_boundary1.shp'
    
    b = open(tiles_list, "r",encoding='UTF-8')
    out_train = b.read()
    file_names = json.loads(out_train)
    
     # Write a new Shapefile
    schema = {
            'geometry': 'Polygon',
            'properties': {'id': 'int'},
            }

    filename,xoff, yoff = file_names[0]
    imgfile = os.path.join(img_path, filename)
    ds1 = gdal.Open(imgfile)
    crs=ds1.GetProjection()
    
    with fiona.open(bound_file, 'w', driver = 'ESRI Shapefile', schema=schema, crs=crs) as c:
        for data in file_names:
            filename,xoff, yoff = data
            imgfile = os.path.join(img_path, filename)
            
            ds = gdal.Open(imgfile)
            gt = ds.GetGeoTransform()
            tilegt=list(gt)
            tilegt[0]=tilegt[0]+xoff*tilegt[1]
            tilegt[3]=tilegt[3]+yoff*tilegt[5]
            
        
            ext = GetExtent(tilegt, 512, 512)
        
            # src_srs = osr.SpatialReference()
            # src_srs.ImportFromWkt(ds.GetProjection())
            # tgt_srs = src_srs.CloneGeogCS()
            # print(tgt_srs)
        
            # crs = tgt_srs.ExportToProj4()
        
            # geo_ext = ReprojectCoords(ext, src_srs, tgt_srs)
    
            poly = Polygon(ext)
        
            # Define a polygon feature geometry with one attribute
        
            
       
            ## If there are multiple geometries, put the "for" loop here
            c.write({
                'geometry': mapping(poly),
                'properties': {'id': 1},
            })