def mfunc(data):
    #import dboxio

    return sum(data)


class mclass(object):
   # import dboxio
   # import rasterio
    import numpy
    def __init__(self):
        from random import SystemRandom
        self.r = SystemRandom()

    def f(self, _):
        x = self.r.random() * 2 - 1
        y = self.r.random() * 2 - 1

        a = x ** 2 + y ** 2
        return 1 if a <= 1 else 0

    def __call__(self, data):
        return sum(map(lambda a: self.f(a), data))


import dboxmr

if __name__ == "__main__":
    # dboxmr.SetHostInfo("127.0.0.1:6600")
    dboxmr.SetHostInfo("10.0.90.63:6600")
    #     dboxmr.SetHostInfo("10.0.81.1:6600")
    n = 20000
    params = [range(n) for _ in range(1000)]
    count = n * len(params)
    #
    #     mc = mclass()
    #     r = [ mfunc( [ mc( para ) for para in params ] ) ]
    #     print("PI =", (float(4 * r[0]) / count))

    r = dboxmr.mapreduce(params, mclass, mfunc, timeout=30)
    print(r)
    print("PI =", (float(4 * r[0]) / count))
