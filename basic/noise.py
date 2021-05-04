import numpy as np
import random

def normal(n, m=0, s=1):
    noise = np.array([random.gauss(m, s) for i in range(n)])
    return noise