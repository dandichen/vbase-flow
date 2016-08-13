import numpy as np
from multiprocessing import Pool
import time
def f(x):
    return x*x

if __name__ == '__main__':
    p = Pool(10)
    a = np.arange(1000000)
    start_time = time.time()
    b = (p.map(f, a))
    end_time = time.time()
    print(end_time - start_time)
    # print b
