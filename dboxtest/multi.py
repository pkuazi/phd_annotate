# params=["/root/data/data1/LC08_L1GT_123032_20200819_20200823_01_T2_B4.TIF",
#         "/root/data/data1/LC08_L1GT_123032_20200819_20200823_01_T2_B5.TIF"]

item=[{'dataid':'LC08_L1GT_123032_20200819_20200823_01_T2_'},
      {'dataid':'LC08_L1TP_121032_20200821_20200905_01_T1_'},
      {'dataid':'LC08_L1TP_121032_20200906_20200906_01_RT_'},
      {'dataid':'LC08_L1TP_122032_20200812_20200822_01_T1_'},
      {'dataid':'LC08_L1TP_122032_20200828_20200905_01_T1_'},
      {'dataid':'LC08_L1TP_122033_20200828_20200905_01_T1_'},
      {'dataid':'LC08_L1TP_123032_20200803_20200807_01_T1_'},
      {'dataid':'LC08_L1TP_123064_20190918_20190926_01_T1_'},
      {'dataid':'LC08_L1TP_123033_20200803_20200807_01_T1_'},
      {'dataid':'LC08_L1TP_123033_20200904_20200904_01_RT_'}
      ]


import dboxmr

def calc(data):
    import dboxio
    import numpy as np
    # obj = __import__("dboxio")
    dataid=data["dataid"]
    # datakey=data['key']
    b4="/root/data/"+dataid+"B4.TIF"
    b5="/root/data/"+dataid+"B5.TIF"
    ds4=dboxio.Open(b4)
    ds5=dboxio.Open(b5)
    extent = ds4.GetTiles()
    band4 = ds4.ReadTile(1,1)
    band5 = ds5.ReadTile(1,1)
    #s = band4 + band5
    #sub = band5 - band4



    # div = sum/sub
    # arr = []
    # for x,y in extent:
    #     band4=ds4.ReadTile(x,y)
    #     band5=ds5.ReadTile(x,y)
    #     sum = band5 + band4
    #     sub = band5 - band4
    #     with np.errstate(divide='ignore', invalid='ignore'):
    #         c = np.true_divide(sub, sum)
    #         c[c == np.inf] = 0
    #         c = np.nan_to_num(c)
    #         arr.append(c)
    #     # div = sub*10000/sum/10000
    #     # arr.append(div)
    return "hello"
#
#
#

if __name__ == "__main__":
    client=dboxmr.Client("10.0.90.63:6600")
    r= client.mapreduce(item,calc,reduce_locally=True,timeout=30,high_priority=False,passive=False)
    print(r)
    # result = calc(item)
    # print(result)

    # a=calc(item)
    # print(a)
