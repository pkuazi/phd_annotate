import itertools
import gdal
# itertools模块提供的全部是处理迭代功能的函数,它们的返回值不是list,而是迭代对象,只有用for循环迭代的时候才真正计算
def gen_tiles_offs(xsize, ysize, BLOCK_SIZE,OVERLAP_SIZE):
    xoff_list = []
    yoff_list = []
    
    cnum_tile = int((xsize - BLOCK_SIZE) / (BLOCK_SIZE - OVERLAP_SIZE)) + 1
    rnum_tile = int((ysize - BLOCK_SIZE) / (BLOCK_SIZE - OVERLAP_SIZE)) + 1
    
    for j in range(cnum_tile + 1):
        xoff = 0 + (BLOCK_SIZE - OVERLAP_SIZE) * j                  
        if j == cnum_tile:
            xoff = xsize - BLOCK_SIZE
        xoff_list.append(xoff)
        
    for i in range(rnum_tile + 1):
        yoff = 0 + (BLOCK_SIZE - OVERLAP_SIZE) * i
        if i == rnum_tile:
            yoff = ysize - BLOCK_SIZE
        yoff_list.append(yoff)
    
    if xoff_list[-1] == xoff_list[-2]:
        xoff_list.pop()    # pop() 方法删除字典给定键 key 及对应的值，返回值为被删除的值
    if yoff_list[-1] == yoff_list[-2]:    # the last tile overlap with the last second tile
        yoff_list.pop()
    
    xoff_list.pop()
    yoff_list.pop()
    print('xoff_list,',xoff_list)
    print('yoff_list', yoff_list)
    
    return [d for d in itertools.product(xoff_list,yoff_list)]
    # itertools.product()：用于求多个可迭代对象的笛卡尔积

def gen_file_list(geotif,BLOCK_SIZE, OVERLAP_SIZE):
    file_list = []
    filename = geotif.split('/')[-1]
    ds = gdal.Open(geotif)
    xsize, ysize = ds.RasterXSize, ds.RasterYSize
    # im = Image.open(geotif).convert('RGB')
    # xsize, ysize = im.size
    off_list = gen_tiles_offs(xsize, ysize, BLOCK_SIZE, OVERLAP_SIZE)
   
    for xoff, yoff in off_list:    
        file_list.append((filename, xoff, yoff))     
    return file_list

if __name__ == '__main__':
    
    xsize = 770
    ysize = 1024
    input_path = '/home/zjh/hz_zjh/gf1c_pms_e120.0_n30.2_20210326_l1a1021729833_area.tif'
    print(gen_file_list(input_path, 1024, 0))
    # x,y = gen_tiles_offs(xsize, ysize, 256,150)
    # print(x)
    # print(y)
    
    