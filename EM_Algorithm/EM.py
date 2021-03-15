
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
from EM_Algorithm.gen_gauss import gen_gauss
from basic.binning import binning
import numpy as np
import matplotlib.pyplot as plt
import math
from sklearn.cluster import KMeans

class EM:
    def __init__(self, data):

