# -*- coding: utf-8 -*-
"""
Created on Tue Apr 27 21:36:58 2021

@author: pine
"""
import scipy.optimize as opt
import math
from sys import platform
import ctypes
import struct
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab
import cv2
import os
from glob import glob
from PIL import Image, ImageEnhance
import pandas as pd
import io
import random
import string


###  2-D Gaussian function with rotation angle
def twoD_Gaussian(xy, amplitude, sigma_x, sigma_y, xo, yo, theta_deg, offset):
    xo = float(xo)
    yo = float(yo)
    theta = theta_deg / 360 * (2 * math.pi)  # in rad
    x = xy[0]
    y = xy[1]
    a = (np.cos(theta) ** 2) / (2 * sigma_x ** 2) + (np.sin(theta) ** 2) / (2 * sigma_y ** 2)
    b = -(np.sin(2 * theta)) / (4 * sigma_x ** 2) + (np.sin(2 * theta)) / (4 * sigma_y ** 2)
    c = (np.sin(theta) ** 2) / (2 * sigma_x ** 2) + (np.cos(theta) ** 2) / (2 * sigma_y ** 2)
    g = offset + amplitude * np.exp(- (a * ((x - xo) ** 2) + 2 * b * (x - xo) * (y - yo)
                                       + c * ((y - yo) ** 2)))
    return g.ravel()

def noise(aoi_size):
    noise = [random.gauss(10,10) for i in range(aoi_size**2)]
    return np.array(noise).reshape(aoi_size, aoi_size)

def get_guess(image_tofit):
    aoi_size = 20
    amp_guess = np.max(image_tofit)
    x_guess = np.argmax(image_tofit) % aoi_size
    y_guess = np.argmax(image_tofit) // aoi_size
    background = np.mean(image_tofit)
    initial_guess = [amp_guess, 10, 10, x_guess, y_guess, 0, background]
    return initial_guess


aoi_size = 20
x = np.array([[i for i in range(20)] for j in range(20)])
y = np.array([[j for i in range(20)] for j in range(20)])

params = [42, 2.3, 2.6, 10.5, 11.6, 10, 6.5]
image_tofit = twoD_Gaussian([x,y], *params).reshape(20,20) + noise(aoi_size)
bounds = ((1, 0.5, 0.5, 0, 0, 0, 0), (255, aoi_size/2, aoi_size/2, aoi_size-1, aoi_size-1, 90, 255))


initial_guess = [80,2,2,5,5,0,0]
initial_guess = get_guess(image_tofit)
popt, pcov = opt.curve_fit(twoD_Gaussian, [x, y], image_tofit.ravel(), initial_guess,
                           bounds=bounds, method='trf')

data_fitted = twoD_Gaussian((x, y), *popt)
fig, ax = plt.subplots(1, 1)
# ax.imshow(image_aoi, cmap=plt.cm.gray, origin='lower',
#           extent=(x.min(), x.max(), y.min(), y.max()))
ax.imshow(image_tofit)

ax.contour(x, y, data_fitted.reshape(aoi_size, aoi_size), 5, colors='r')



