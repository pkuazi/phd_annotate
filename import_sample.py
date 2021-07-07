

import json
import uuid
import os
import psycopg2

import rasterio.features
import rasterio.warp

from osgeo import ogr

from osgeo import osr


filepath = '/mnt/chicago_tiles/tif'

geojson_filepath = '/mnt/chicago_tiles/geojson'

conn=psycopg2.connect(database="laymark",user="postgres",password="",host="10.0.85.85",port="5432")
print("conn success")
cur=conn.cursor() 



def get_uuid():
	return str(uuid.uuid4()).replace("-","")


def get_files(filepath):
	files = os.listdir(filepath)
	# files = map(lambda filename : '{}/{}'.format(filepath,filename),files)
	return files

def reproject_polygon(geom, crs_proj4_src, crs_proj4_dst, fmt="geojson"):

     source = osr.SpatialReference()

     source.ImportFromProj4(crs_proj4_src)

     target = osr.SpatialReference()

     target.ImportFromProj4(crs_proj4_dst)

     transform = osr.CoordinateTransformation(source, target)

     if fmt == "wkt":

         obj = ogr.CreateGeometryFromWkt(geom)

         obj.Transform(transform)

         return obj.ExportToWkt()

     if fmt == "geojson":

         obj = ogr.CreateGeometryFromJson(geom)

         obj.Transform(transform)

         return obj.ExportToJson()

     return None


def create_subtask(staskid,extent):
	# staskid = get_uuid()
	select_subtask_sql = "select * from mark_subtask where guid='{}';".format(staskid)
	cur.execute(select_subtask_sql)
	res = cur.fetchone()
	if res:
		return
	create_subtask_sql = "insert into mark_subtask (guid,taskid,state,reward,num_currentuser,num_maxuser,geojson) values ('{guid}','679a5dd7d89140eebb75bf723e256f67','1',100,0,3,'{geojson}');".format(guid=staskid,geojson=json.dumps(extent))
	cur.execute(create_subtask_sql)
	conn.commit()



def create_sample(staskid):
	file = "{}/softmax_prob_{}.geojson".format(geojson_filepath, staskid)

	with open(file,'r') as fp:
		data = json.loads(fp.read())

	features = data.get("features")

	for feature in features:
		properties = feature.get("properties",{})
		id = properties.get("id")
		if id == 999 or id  == 9999:
			continue
		geojson = json.dumps(feature.get("geometry"))
# 		geojson = reproject_polygon(geojson,'+init=epsg:26916','+init=epsg:4326')
		guid = get_uuid()
		create_sample_sql = "insert into mark_sample (guid,taskid,ftype,state,tagid,userid,geojson) values ('{guid}','{staskid}','3','1','0','9','{geojson}');".format(guid=guid,staskid=staskid,geojson=geojson)
		cur.execute(create_sample_sql)
		conn.commit()


def get_sample_extent(filename):
	tif = "{}/{}".format(filepath,filename)
	
	with rasterio.open(tif) as dataset:

		# Read the dataset's valid data mask as a ndarray.
		mask = dataset.dataset_mask()

		# Extract feature shapes and values from the array.
		geom, val = next(rasterio.features.shapes(mask, transform=dataset.transform))
		
		extent = rasterio.warp.transform_geom(dataset.crs, 'EPSG:4326', geom, precision=6)

		return extent

		# for geom, val in rasterio.features.shapes(mask, transform=dataset.transform):

		# 	# Transform shapes from the dataset's own coordinate
		# 	# reference system to CRS84 (EPSG:4326).
		# 	geom = rasterio.warp.transform_geom(dataset.crs, 'EPSG:4326', geom, precision=6)

		# 	# Print GeoJSON shapes to stdout.
		# 	print(geom)

if __name__ == "__main__":
	import time
	start_time = time.time()
	files = get_files(filepath)
	i = 0
	for file in files:
		# fileos.path.basename(file)
		staskid = file.split(".")[0]
		print(staskid)

# 		extent = get_sample_extent(file)
# 		create_subtask(staskid,extent)

		create_sample(staskid)
		i += 1
		print(i)
	print(time.time() - start_time)