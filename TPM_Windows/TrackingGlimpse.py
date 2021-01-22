# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:30:44 2020
version: 20201111
use gaussian fit to center each point
version: 20201211
use two classed to store data information

@author: YCC

Read one image of glimpse file
"""

### setting parameters
path_folder = r'C:\Users\hwlab\Desktop\17-550bp-30ms-33fps'

size_tofit = 20
read_mode = 0 # mode = 0 is only calculate 'frame_setread_num' frame, other numbers(default) present calculate whole glimpsefile 
frame_setread_num = 20 # only useful when mode = 0, can't exceed frame number of a file

fit_mode = 'multiprocessing' #'multiprocessing'
path_mode = 'm' # 'a': auto-pick cd, 'm': manually select

criteria_dist = 5 # beabs are closer than 'criteria_dist' will remove
aoi_size = 20
frame_read_forcenter = 2 # no need to change, frame to autocenter beads
N_loc = 5
contrast = 10
low = 40
high = 120 
blacklevel = 240
### preparing input parameters
settings = [criteria_dist, aoi_size, frame_read_forcenter, N_loc, contrast, low, high, blacklevel]
### import used modules first
import scipy.optimize as opt
import math
import multiprocessing as mp
import ctypes
import struct
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import cv2
import os
from glob import glob
import datetime
from PIL import Image,ImageEnhance
from multiprocessing import freeze_support
import csv
import time
# from Localization import *

### getting parameters
# fps = 100
today = datetime.datetime.now()
filename_time = str(today.year)+str(today.month)+str(today.day)
if path_mode == 'm':
    # file_folder = input('type your folder path ')
    file_folder = path_folder
else:
    file_folder = os.getcwd()





### define a class for image info. e.g. path, header...
class data_folder:
    def __init__(self, path_folder):
        self.path_folder = os.path.abspath(path_folder)
        self.path_header = os.path.abspath(path_folder + '\header.glimpse')
        self.path_header_utf8 = self.path_header.encode('utf8')
        self.path_data = [x for x in glob(self.path_folder + '/*.glimpse') if x != self.path_header ]
        self.header = []
        ### get file info.
        self.size_a_image = 0  # 8bit format default
        self.data_type = ''
        self.file_size = [] 
        self.frame_per_file = []
        self.frame_total = 0 # cal frame number of this file
        self.info = []
    def getheader(self):
        mydll = ctypes.windll.LoadLibrary('./GetHeader.dll')
        GetHeader = mydll.ReadHeader  # function name is ReadHeader
        # assign variable first (from LabVIEW)
        # void ReadHeader(char String[], int32_t *offset, uint8_t *fileNumber, 
        # uint32_t *PixelDepth, double *timeOf1stFrameSecSince1104 (avg. fps (Hz)),uint32_t *Element0OfTTB, 
        # int32_t *RegionHeight, int32_t *RegionWidth, 
        # uint32_t *FramesAcquired)
        # ignore array datatype in header.glimpse
        GetHeader.argtypes = (ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint),
                      ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_double), ctypes.POINTER(ctypes.c_uint),
                      ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
                      ctypes.POINTER(ctypes.c_uint))
        offset = ctypes.c_int(1)
        fileNumber = ctypes.c_uint(1)
        PixelDepth = ctypes.c_uint(1)
        Element0OfTTB = ctypes.c_uint(1)    
        timeOf1stFrameSecSince1104 = ctypes.c_double(1)
        RegionHeight = ctypes.c_int(1)
        RegionWidth = ctypes.c_int(1)
        FramesAcquired = ctypes.c_uint(1)
        
        GetHeader(self.path_header_utf8, offset, fileNumber, 
                  PixelDepth, timeOf1stFrameSecSince1104, Element0OfTTB, 
                  RegionHeight, RegionWidth, 
                  FramesAcquired) # There are 8 variables.
        self.header = [FramesAcquired.value, RegionHeight.value, RegionWidth.value, 
            PixelDepth.value, offset.value, fileNumber.value, 
            Element0OfTTB.value, timeOf1stFrameSecSince1104.value]
    
    def getdatainfo(self):       
        ### get file info.
        if self.header[3] == 0: # 8 bit integer
            self.data_type = 'B'
            pixel_depth = 1
        else:
            self.data_type = 'h'
            pixel_depth = 2
        self.size_a_image = self.header[1] * self.header[2] * pixel_depth  # 8bit format default
        self.file_size = [Path(x).stat().st_size for x in self.path_data] 
        self.frame_per_file = [int(x/self.size_a_image) for x in self.file_size]
        self.frame_total = sum(self.frame_per_file) # cal frame number of this file
        self.info = [self.header, self.frame_per_file, self.path_data, self.data_type, self.size_a_image]
### define a class for all glimpse data
class Gimage:
    def __init__(self, info, criteria_dist = 10, aoi_size = 20):
        ##  info: [[header], [frame_per_file], [path_data], data_type, size_a_image]
        self.header = info[0]
        self.frame_per_file = info[1]
        self.path_data = info[2]
        self.data_type = info[3]
        self.size_a_image = info[4]
        # self.frame = frame_i
        self.height = self.header[1]
        self.width = self.header[2]
        
        self.read1 = [] # one image at i
        self.readN = [] # N image from i
        self.offset = []
        self.fileNumber = []
        self.contours = []
        self.edges = []
        self.cX = []
        self.cY = []
        self.perimeters = []
        self.areas = []
        self.radius_save = []
        self.criteria = criteria_dist
        self.AOI_size = aoi_size
        self.image = [] # image used to process
        self.intensity = []
        self.image_cut = []
        self.AOIimage = []

    
    ##  get offset array
    def getoffset(self):
        self.size_a_image = self.header[1] * self.header[2]
        frame_total = sum(self.frame_per_file)
        frame_file_max = self.frame_per_file[0]
        offset = []
        fileNumber = []
        a = 0
        b = 0
        for i in range(frame_total):
            offset += [a * self.size_a_image]
            fileNumber += [np.floor(i/frame_file_max).astype(int)]
            if np.floor((i+1)/frame_file_max) == b:
                a += 1
            else:
                a = 0
                b += 1
        self.offset = offset
        self.fileNumber = fileNumber
                  
    ##  read one image at frame_i (0,1,2,...,N-1)
    def readGlimpse1(self, frame_i = 0):
        fileNumber = self.fileNumber[frame_i]
        offset = self.offset[frame_i]
        with open(self.path_data[fileNumber],'rb') as f:
            f.seek(offset)
            data = f.read(self.size_a_image)    
            decoded_data = struct.unpack('>' + str(self.size_a_image * 1) + self.data_type, data)
            self.read1 = np.reshape(decoded_data, (self.height, self.width) )
            self.image = self.read1
        return self.read1
    
    ##  read N image from frame_i (0,1,2,...,N-1)
    def readGlimpseN(self, frame_i = 0, N = 50):
        fileNumber = self.fileNumber[frame_i : frame_i+N]
        offset_toread = [self.offset[x] for x in set(fileNumber)]
        path_toread = [self.path_data[x] for x in set(fileNumber)]
        frame_toread = [sum(fileNumber == x) for x in set(fileNumber)]
        decoded_data = []
        for path, frame, offset in zip(path_toread, frame_toread, offset_toread):
            with open(path,'rb') as f:
                f.seek(offset)
                data = f.read(self.size_a_image * frame)
                decoded_data += struct.unpack('>' + str(self.size_a_image * frame) + self.data_type, data)
        self.readN = np.reshape(decoded_data, (N, self.height, self.width) )
        return self.readN
    
    ##  load image
    def loadimage(self, image):    
            self.image = image
    
    ##  stack multiple images    
    def stackimageN(self, imageN):
        return np.mean((imageN.T), 2).T.astype('uint8')

    ##  enhance contrast
    def contrast(self, contrast = 1):
        enh_con = ImageEnhance.Contrast(Image.fromarray(self.image))
        image_contrasted = enh_con.enhance(contrast)
        self.image = np.array(image_contrasted)
        return self.image
        
    
    ##  get egde using frame in file,f
    def getContour(self, low = 30, high = 90): 
        ##  get edges using openCV
        self.image_cut = np.uint8(self.image[0+18:self.height-18, 0+32:self.width-32])
        self.edges = cv2.Canny(self.image_cut, low, high) # cv2.Canny(image, a, b), reject value < a and detect value > b
        # ret, thresh = cv2.threshold(self.edges, 0, 50, cv2.THRESH_BINARY) # THRESH_BINARY: transform into black/white depending on low/high value
        self.contours, hierarchy = cv2.findContours(self.edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    ##  get center point using moment of edge
    def getXY(self):
        radius = []
        n_contours = len(self.contours)
        for i in range(n_contours):
            c = self.contours[i]
            self.perimeters += [cv2.arcLength(c, True)]
            self.areas += [cv2.contourArea(c)]
            if (self.perimeters[-1] <= 6) | (self.areas[-1] <= 2) | (len(c) < 2):
                continue
            radius += [self.areas[-1]/self.perimeters[-1]]
            M = cv2.moments(c)
            #(1 < radius[-1] < 5) & (M['m10'] != 0) & (M['m01'] != 0)
            if (M['m00'] != 0):            
                self.radius_save += [radius[-1]]
                self.cX +=  [(M['m10'] / M['m00'])+32]
                self.cY +=  [(M['m01'] / M['m00'])+18]   

    ## remove beads are too close, choose two image, refer to smaller bead#
    def removeXY(self): 
        cX1 = np.array(self.cX) # len of cXr1 is smaller, as ref
        cY1 = np.array(self.cY)
        i_dele = np.empty(0).astype(int)
        for i in range(len(cX1)):
            dx = cX1 - cX1[i]
            dy = cY1 - cY1[i]
            dr = np.sqrt(dx**2 + dy**2)
            # i_dr = np.argsort(dr)
            # i_XYr_self = i_dr[0]
            if any(dr[dr != 0] <= self.criteria):
                i_dele = np.append(i_dele, int(i))                
        self.cX = np.delete(cX1, i_dele)
        self.cY = np.delete(cY1, i_dele)
        self.radius_save = np.delete(self.radius_save, i_dele)      
        
    ##  get intensity of AOI
    def getintensity(self, aoisize = 20): # i: bead number: 1,2,3,...,N
        half_size = int(aoisize/2)    
        for i in range(len(self.cX)):
            horizontal = int(self.cY[i]) # width
            vertical = int(self.cX[i])   # height
            self.intensity += [np.mean(self.image[horizontal-half_size:(horizontal+half_size), vertical-half_size:(vertical+half_size)])] # [x,y] = [width, height]
        self.intensity = np.array(self.intensity) 
        
    ##  remove low intensity
    def removeblack(self, blacklevel = 150):
        i_dele = np.empty(0).astype(int)
        for i in range(len(self.cX)):
            if self.intensity[i] < blacklevel:
                i_dele = np.append(i_dele, int(i))
        self.cX = np.delete(self.cX, i_dele)
        self.cY = np.delete(self.cY, i_dele)
        self.radius_save = np.delete(self.radius_save, i_dele)
        self.intensity = np.delete(self.intensity, i_dele)
        
    ##  plot X,Y AOI in given image
    def drawAOI(self):
        n = len(self.cX)
        for i in range(n):
            cv2.circle(self.image, (int(self.cX[i]), int(self.cY[i])), self.AOI_size, (0, 0, 0), 1)
            cv2.putText(self.image, str(i+1), (int(self.cX[i]+10), int(self.cY[i]+10))
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 1, cv2.LINE_AA)
        return self.image

        
        
### define main() for localize all beads in a frame
def localize(path_folder, settings = [5,20,0,10,10,30,90,230]): # four steps, getContour => getXY => removeXY => plotXY
    ### get parameters, settings = [criteria_dist, aoi_size, frame_read_forcenter, N, contrast, low, high, blacklevel]
    criteria_dist, aoi_size, frame_read_forcenter, N, contrast, low, high, blacklevel = settings
    
    ### pick one image to localize bead position

    print('opening file...')
    # with open(file_name[0], 'rb') as f:
    folder = data_folder(path_folder)
    folder.getheader()
    folder.getdatainfo()
    info = folder.info
    Gdata = Gimage(info, criteria_dist, aoi_size)
    Gdata.getoffset()
    imageN = Gdata.readGlimpseN(frame_read_forcenter, N)
    image = Gdata.stackimageN(imageN)
    Gdata.loadimage(image)
    print('start centering')
    
    Gdata.contrast(contrast = 10)
    Gdata.getContour(low, high)
    Gdata.getXY()
    Gdata.removeXY()    
    Gdata.getintensity(aoisize = 2)
    Gdata.removeblack(blacklevel = 220)
    Gdata.drawAOI()
    print('finish centering')
    return Gdata, folder

##  2-D Gaussian function with rotation angle
def twoD_Gaussian(xy, amplitude, xo, yo, sigma_x, sigma_y, theta_rad, offset):
    xo = float(xo)
    yo = float(yo)
    theta = theta_rad/360*(2*math.pi) # in degree
    x = xy[0]
    y = xy[1]    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                                    + c*((y-yo)**2)) )
    return g.ravel()

### get image of certain AOI
def getAOI(image, row, col, size=20):
    row = int(row) # cY, height
    col = int(col)   # cX, width
    size_half = int(size/2)
    image_cut = image[row-size_half:(row+size_half), col-size_half:(col+size_half)]
    return image_cut
        
###  tracking bead position in a image, get center and std of X,Y using Gaussian fit
def trackbead(image):  
    xc = []
    yc = []
    sx = []
    sy = []
    para_fit = []
    for j in range(bead_number):
        image_tofit = getAOI(image, aoi[1][j], aoi[0][j])
        ## enhance contrast
        # image_bead = image[horizontal-10:(horizontal+10), vertical-10:(vertical+10)] # [x,y] = [width, height]
        # image_bead_bur = cv2.GaussianBlur(image_bead, (5, 5),2,2)
        ## increase contrast
        # enh_con = ImageEnhance.Contrast(Image.fromarray(image_bead_bur))
        # contrast = 10
        # image_contrasted = enh_con.enhance(contrast)
        # image_tofit = np.array(image_contrasted)
        popt=[]
        pcov=[]
        try:
            popt, pcov = opt.curve_fit(twoD_Gaussian, [x, y],image_tofit.ravel(), parameters[j])
            # popt: optimized parameters, pcov: covariance of popt, diagonal terms are variance of parameters
            # data_fitted = twoD_Gaussian((x, y), *popt)
            xc += [popt[1]] # x position of each frame
            yc += [popt[2]] #
            sx += [popt[3]]
            sy += [popt[4]]
            para_fit += [list(popt)]
            parameters[j] = list(popt)
        except RuntimeError:
            # popt, pcov = opt.curve_fit(twoD_Gaussian, [x, y],image_tofit.ravel(), initial_guess)
            xc += [0] # x position of each frame
            yc += [0] #  
            sx += [0]
            sy += [0]
            para_fit += [0,0,0,0,0,0,0]
            parameters[j] = [initial_guess] # initial guess for all beads
    # return [xc, yc, sx, sy, para_fit]
    return para_fit

### get parameters for trackbead fitting
def preparefit_info(read_mode): 
    x = np.array([[i for i in range(20)] for j in range(20)]).astype(float)
    y = np.array([[j for i in range(20)] for j in range(20)]).astype(float)
    aoi = [Gdata.cX, Gdata.cY]
    bead_number = len(aoi[0])
    initial_guess = [30,11,11,2,2,0,127]
    parameters = [initial_guess] * bead_number
    if read_mode == 0:
        N = frame_setread_num
    else:
        N = folder.frame_total
    
    return x, y, aoi, bead_number, initial_guess, parameters, N

### get parameters for each fitting loop
def getloop_info(frame_start, frame_total, size_tofit):

    frame_i = []
    N = []
    i_run = np.ceil(frame_total/size_tofit).astype(int)
    for i in range(i_run):
        frame_i += [ min((i)*size_tofit, frame_total) ]
        if frame_total>(i+1)*size_tofit:
            N += [size_tofit]
        else:
            N += [frame_total-(i)*size_tofit]
    return frame_i, N # frame_i: frame_start for each loop, N: size to fit for each loop

### get csv writting info
def getcsvinfo(bead_number):
    bead_namex = []
    bead_namey = []
    bead_namesx = []
    bead_namesy = []
    ##   create csv file to store xy position data
    for i in range(bead_number):
        bead_namex += ['beadx '+str(i+1)]
        bead_namey += ['beady '+str(i+1)]
        bead_namesx += ['stdx '+str(i+1)]
        bead_namesy += ['stdy '+str(i+1)]
    bead_namexy = bead_namex + bead_namey + bead_namesx + bead_namesy
    return bead_namexy

def print_start(frame_i, frame_f):
    print('start analyzing ' + 'frame: ' + str(frame_i) + ' to ' + str(frame_f) )

def print_finish(frame_i, frame_f, frame_total):
    print('finish analyzing ' + 'frame: ' + str(frame_i) + ' to ' + str(frame_f) + ', progress: ' + 
                          str(int(100*frame_f/frame_total)) + '%')

### not finishing for choosing multi or single core
# def fit_mode(image_eachframe, fit_mode = 'multiprocessing'):
#     if fit_mode == 'multiprocessing':
#         with mp.Pool(mp.cpu_count()-4) as pool:
#             r = pool.map(trackbead, image_eachframe)
#             result.append(r)
#             if len(result) == len(frame_i):
#                 for k in range(len(result)):
#                     for m in range(len(result[k])):
#                         writer.writerow(result[k][m][0] + result[k][m][1] + result[k][m][2] + result[k][m][3])
#                 print_finish(i, i+n, N)
#                 print('saving...')
#             else:
#                 print_finish(i, i+n, N)
#     else:
#         r = []
#         for image in image_eachframe:
#             r += trackbead(image)
#     return r
    


### main to tracking each beads over each frame
def fit_all_frame(Gdata, frame_start, N, size_tofit):
    frame_i, frame_tofit = getloop_info(frame_start, N, size_tofit)
    bead_namexy = getcsvinfo(len(Gdata.cX))
    result = []
    ##  open .csv to be saved
    with open(file_folder+'/'+filename_time+'-xy and sigma xy.csv', 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        writer.writerow(bead_namexy)
        for i, n in zip(frame_i, frame_tofit):
            image_eachframe = Gdata.readGlimpseN(i, n)
            print_start(i, i+n)
            # r = fit_mode(image_eachframe, fit_mode = fit_mode)
            # result.append(r)
            with mp.Pool(mp.cpu_count()-4) as pool:
                # freeze_support()
                r = pool.map(trackbead, image_eachframe)
                # pool.close()
                # pool.join()
                result += r
                if len(result) == N:
                    data = np.array(result)
                    for k in range(len(result)): # number of frame
                    ##  save x,y,sx,sy 
                        writer.writerow(list(data[k][:,1]) + list(data[k][:,2]) + list(data[k][:,3]) + list(data[k][:,4]))
                    print_finish(i, i+n, N)
                    print('saving...')
                else:
                    print_finish(i, i+n, N)
    return result, r



### main to get localization information
Gdata, folder = localize(path_folder, settings)
image_localization = Gdata.image
plt.figure()
plt.imshow(image_localization, cmap='gray', vmin=0, vmax=255)
cv2.imwrite(os.path.abspath(path_folder) + '/output.png', image_localization)

##  initialize fitting parameters
x, y, aoi, bead_number, initial_guess, parameters, N = preparefit_info(read_mode)

##  start multi-processing using .map()
t1 = time.time()
if __name__ == '__main__': # for windows, lacking fork
    result, r = fit_all_frame(Gdata, frame_start=0, N = N, size_tofit = size_tofit)
time_spent = time.time() - t1
print('spent ' + str(time_spent) + ' s')


