import numpy as np
import random

def normal(n, m=0, s=1):
    noise = np.array([random.gauss(m, s) for i in range(n)])
    return noise

def normal_2d(aoi_size, m=0, s=1):
    noise = [random.gauss(m, s) for i in range(aoi_size**2)]
    return np.array(noise).reshape(aoi_size, aoi_size)