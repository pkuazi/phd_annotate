#
# -*-coding:utf-8 -*-
#
# @Author: zhaojianghua
# @Date  : 2020-08-27 09:12
#
import os
import numpy as np
import pandas as pd
import fiona
import rasterio
from geotrans import GeomTrans
from maskimage import mask_feats_by_geometry
from sklearn.ensemble import RandomForestClassifier
import pickle,time
import cv2

def read_imgfeature_bands(imagepath, bandnames):
    featarray = pd.DataFrame() 
    files = os.listdir(imagepath)
    img_dict = {}
    for file in files:
        for b in bandnames:
            if file.endswith(b + '.TIF'):
                file = os.path.join(imagepath, file)
                raster = rasterio.open(file, 'r')
                gt = raster.get_transform()
                crs = raster.crs.to_wkt()
                array = raster.read(1)
                img_dict[b] = array
    img_dict['gt'] = gt
    img_dict['crs'] = crs
    return img_dict

                
def generate_training_data(image_col, samples_file, property):
    keys = image_col.keys()
    bands = []
    for b in keys:
        if b != 'gt' and b != 'crs':
            bands.append(b)
    
    vector = fiona.open(samples_file, 'r')
    sample_proj = vector.crs_wkt
  
    dst_proj = image_col['crs']
#     dst_proj = image_col['crs'].to_wkt()
    print(sample_proj, dst_proj)
    feat = []
    train_y = []
    
    for feature in vector:
        if 'class_id' in feature['properties'].keys():
            class_id = feature['properties'][property]
            train_y.append(class_id)
        else:
            print("no class id...")
            return None
        geojson = feature['geometry']
        if geojson['type'] == 'Point':
            x = geojson['coordinates'][0]
            y = geojson['coordinates'][1]
        x_imgcrs, y_imgcrs = GeomTrans(sample_proj, dst_proj).transform_point([x, y])
        
        #     [321285.0, 30.0, 0.0, 4424415.0, 0.0, -30.0]
        row = int((y_imgcrs - image_col['gt'][3]) / image_col['gt'][5])
        col = int((x_imgcrs - image_col['gt'][0]) / image_col['gt'][1])
        
        for b in keys:
            if b != 'gt' and b != 'crs':
                data = image_col[b]
                if 0 < col < data.shape[0] and 0 < row < data.shape[1]:
                    feat_value = data[row][col]
                    feat.append(feat_value)
#                     print("the feature value is :", feat_value)
            else:
                continue
    band_num = len(bands)
    feat = np.array(feat)
    train_x = feat.reshape(-1, band_num)
    
    X = pd.DataFrame(train_x)
    Y = pd.DataFrame(train_y)
#     print(X)
    train_df = pd.concat([X, Y], axis=1)
    bands.append(property)
    train_df.columns = bands
#     result.to_csv(dst_file)

    print(train_df)
    return train_df   

def Training(image, samples_file, classProperty,features,model_npy,options):
#     classifier = ee.Classifier.smileRandomForest(10).train(training, classProperty, features);
#     var training = image.sampleRegions({
#   collection: samples,
#   properties: [classProperty],
#   scale: 30
# });
    '''
    'options':n_estimators,'samples_file':os.path.join(samples_path, 'samples_%s.geojson' % (array_date)), 'classProperty':'class_id', 'features':['B1', 'B2', 'B3', 'B4', 'B5', 'B6', 'B7'], 'image':imagepath
    '''
#         train a model
    xa_img_dict = read_imgfeature_bands(image, features)
    xa_train_df = generate_training_data(xa_img_dict, samples_file, classProperty)
    
    training_X = xa_train_df.iloc[:, :-1]
    training_y = xa_train_df.iloc[:, -1]
    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=9, random_state=0)
    clf.fit(training_X, training_y)
     
    pickle.dump(clf, open(model_npy, 'wb'))

def Predict(nparr, model) :
    '''
    nparr 256 * 256 * feature_num
    options : {'array_date':20080327}
    ''' 
    row, col = nparr.shape[:2]
    feat = []
    for i in range(row):
        for j in range(col):
            feat.append(nparr[i][j])
    feat = np.array(feat)
    df = pd.DataFrame(feat)
    
    clf = pickle.load(open(model, 'rb'))
    r = clf.predict(df)
    classified_array = r.reshape(row, col)
    classified_array[classified_array == 6] = 2
    classified_array[classified_array == 7] = 3
#         1 zhibei;2 qita; 3 luwei; 4 buildings; 5 water; 
    return classified_array
def object_supervised_classification():
    return

def main():
    feature_img = '/tmp/ft_from_layers/deeplab_lastlayer.tif'
    labeled_polys = '/tmp/ergc/ergc_plys_2_10_.shp'
    segments_csv = '/tmp/ergc/ergcorigin_00.csv'
    
    Feat_raster = rasterio.open(feature_img, 'r')
    
    
    seg_df=pd.read_csv(segments_csv, sep=',',header=None)
    segments_mask=seg_df.values
    
    
    #save osm_id and raster features into csv
    train_df = pd.DataFrame()
    test_df = pd.DataFrame()
        # df = df.append(pd.DataFrame({'osm_id':geomid,'feats':feats},index = [i]),ignore_index=True) 

    vector = fiona.open(labeled_polys, 'r')
    vec_proj = vector.crs_wkt
    for feature in vector:
        feature_id = feature['properties']['id']
        label=feature['properties']['label']
        
        geojson = feature['geometry']
    
    # if multipolygon, split it into polygons
        if geojson['type']=='MultiPolygon':
            num_plys=len(geojson['coordinates'])
            for i in range(num_plys):
                geomid=feature_id+'_%s'%i
                geojson = feature['geometry']
                poly_coords=geojson['coordinates'][i][0]
                feats, deltat= mask_feats_by_geometry(poly_coords,vec_proj, Feat_raster)
                if feats is None:
                    continue
                feats.append(label)
                if label!=9999:
                    train_df[geomid]=feats
                else:
                    test_df[geomid]=feats

        elif geojson['type']=='Polygon':
            poly_coords=geojson['coordinates'][0]
            feats, deltat= mask_feats_by_geometry(poly_coords,vec_proj, Feat_raster)
            if feats is None:
                continue
            feats.append(label)
            if label!=9999:
                train_df[feature_id]=feats
            else:
                test_df[feature_id]=feats

    train_df=train_df.transpose()
    test_df=test_df.transpose()

    models_path='/tmp'
    # Random Forest
    timest= time.ctime().split(' ')[4]+'-'+time.ctime().split(' ')[1]+'-'+time.ctime().split(' ')[2]+'-'+time.ctime().split(' ')[3]
    model_npy = os.path.join(models_path, 'random_forest_%s.sav' % (timest)) 
    if os.path.exists(model_npy):  
        clf = pickle.load(open(model_npy, 'rb'))
    else:
        print('no model...')
    
    training_X = train_df.iloc[:, :-1]

    training_y = train_df.iloc[:, -1]

    clf = RandomForestClassifier(n_estimators=10, max_depth=None, min_samples_split=9, random_state=0)
    clf.fit(training_X, training_y)
     
    pickle.dump(clf, open(model_npy, 'wb'))
    
    clf = pickle.load(open(model_npy, 'rb'))
    testing_X=test_df.iloc[:,:-1]
    testing_Y = clf.predict(testing_X)
    
    
    # test_df[:,-1]= testing_Y
        # Find the name of the column by index
    n = test_df.columns[-1]
    
    # Drop that column
    test_df.drop(n, axis = 1, inplace = True)
    
    # Put whatever series you want in its place
    test_df[n] = testing_Y
    
    df = train_df.append(test_df)
    print(df.shape)
    print('output the final image......')
    rf_seg=np.full(segments_mask.shape, 9999, dtype=int)

    num = segments_mask.max()+1
    for i in range(num):
        label_col_idx = df.shape[1]-1
        label=df.loc[i,label_col_idx]
        rf_seg[segments_mask==i]=int(label)
    
    proj=Feat_raster.crs.wkt
    gt = Feat_raster.transform
    
    with rasterio.open("/tmp/rf_classified_mean_std.tif", 'w', driver='GTiff', width=rf_seg.shape[1], height=rf_seg.shape[0], crs=Feat_raster.crs,
                       transform=Feat_raster.transform, dtype=np.uint16, nodata=256,count=1) as dst:
        # Write the src array into indexed bands of the dataset. If `indexes` is a list, the src must be a 3D array of matching shape. If an int, the src must be a 2D array.
        dst.write(rf_seg.astype(rasterio.uint16),1)
    

if __name__ == '__main__': 
    main()  
    
