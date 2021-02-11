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
path_folder = r'F:\YCC\20210205\4-1895bp\1-200ms-440uM_BME_gain20'

# size_tofit = 10
read_mode = 1 # mode = 0 is only calculate 'frame_setread_num' frame, other numbers(default) present calculate whole glimpsefile 
frame_setread_num = 20 # only useful when mode = 0, can't exceed frame number of a file

# fit_mode = 'multiprocessing' #'multiprocessing'
path_mode = 'auto' # 'a': auto-pick cd, 'm': manually select
if path_mode == 'm':
    path_folder = input('type your folder path ')
    
criteria_dist = 10 # beabs are closer than 'criteria_dist' will remove
aoi_size = 20
frame_read_forcenter = 0 # no need to change, frame to autocenter beads
N_loc = 40 # number of frame to stack and localization
contrast = 11
low = 40
high = 120
blacklevel = 40

### preparing input parameters
# settings = [criteria_dist, aoi_size, frame_read_forcenter, N_loc, contrast, low, high, blacklevel]
### import used modules first
import scipy.optimize as opt
import math
from sys import platform
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
import pandas as pd


### define a class for all glimpse data
class BinaryImage:
    def __init__(self, path_folder, read_mode = 0, frame_setread_num = 10, 
                 criteria_dist = 10, aoi_size = 20, frame_read_forcenter = 0,
                 N_loc = 20, contrast = 10, low = 50, high = 150,
                  blacklevel = 50):
        self.path_folder = os.path.abspath(path_folder)
        self.path_header = os.path.join(path_folder, 'header.glimpse')
        self.path_header_utf8 = self.path_header.encode('utf8')
        self.path_data = [x for x in sorted(glob(self.path_folder + '/*.glimpse')) if x != self.path_header ]
        [self.frames_acquired, self.height, self.width, self.pixeldepth, self.avg_fps] = self.getheader()
        [self.frame_per_file, self.path_data, self.data_type, self.size_a_image] = self.getdatainfo()
        self.time_axis = np.arange(0, self.frames_acquired)/self.avg_fps

        self.criteria_dist = criteria_dist
        self.aoi_size = aoi_size
        self.frame_read_forcenter = frame_read_forcenter
        self.N_loc = N_loc
        self.contrast = contrast
        self.low = low
        self.high = high
        self.blacklevel = blacklevel
        self.offset, self.fileNumber = self.getoffset()
        
        self.read1 = [] # one image at i
        self.readN = [] # N image from i
        # self.contours = []
        # self.edges = []
        self.image = [] # image used to process
        self.cX = []
        self.cY = []
        self.perimeters = []
        self.areas = []
        self.radius_save = []
        # self.intensity = []
        # self.image_cut = []
        # self.AOIimage = []
        self.x_fit = []
        self.y_fit = []
        self.bead_number = 0
        self.initial_guess = []
        self.parameters = []
        self.N = 0
        
        
     
###########################################################################        
    ##  main for localization
    def Localize(self):
        print('start centering')
        imageN = self.readGlimpseN(self.frame_read_forcenter, self.N_loc)
        image = self.stackimageN(imageN)
        image = self.enhance_contrast(image, self.contrast)
        contours = self.getContour(image, self.low, self.high)
        cX, cY = self.getXY(contours)
        cX, cY = self.removeXY(cX, cY, self.criteria_dist)
        intensity = self.getintensity(image, cX, cY, self.aoi_size)
        cX, cY = self.removeblack(cX, cY, intensity, blacklevel)
        self.cX = cX
        self.cY = cY
        self.bead_number = len(cX)
        image = self.drawAOI(image, cX, cY, self.aoi_size)
        self.show_grayimage(image, save = True)
        # self.x_fit, self.y_fit, self.bead_number, self.initial_guess, self.parameters, self.N = self.preparefit_info(
        #     read_mode, frame_setread_num, self.frames_acquired, self.bead_number, self.aoi_size)
        print('finish centering')
        return image, cX, cY
    
    ##  main for tracking all frames and all beads(cX, cY)
    def Track_All_Frames(self, read_mode, frame_setread_num):
        # read_mode = self.read_mode
        # frame_setread_num = self.frame_setread_num
        cX = self.cX
        cY = self.cY
        self.x_fit, self.y_fit, self.initial_guess, self.parameters, self.N = self.preparefit_info(
            read_mode, frame_setread_num, self.frames_acquired, self.aoi_size)
        p0 = np.array(self.parameters)
        result = []
        for i in range(self.N):
            image = self.readGlimpse1(i)
            data, parameters = self.trackbead(image, cX, cY, self.aoi_size, i+1, p0)
            p0 = self.update_p0(p0, parameters, i+1) # update fitting initial guess
            result += data
            print(f'frame {i+1}')
        result_saved = np.array(result)
        return result_saved, self.time_axis
    

###############################################################################
    ### methods for localization
    ##  stack multiple images    
    def stackimageN(self, imageN):
        return np.mean((imageN.T), 2).T.astype('uint8')

    ##  load image
    def loadimage(self, image):    
            self.image = image

    ##  enhance contrast
    def enhance_contrast(self, image, contrast = 10):
        enh_con = ImageEnhance.Contrast(Image.fromarray(image))
        image_contrasted = enh_con.enhance(contrast)
        image = np.array(image_contrasted)
        return image
        
    
    ##  get egde using frame in file,f
    def getContour(self, image, low = 30, high = 90): 
        ##  get edges using openCV
        image_cut = np.uint8(image[0+18:self.height-18, 0+32:self.width-32])
        edges = cv2.Canny(image_cut, low, high) # cv2.Canny(image, a, b), reject value < a and detect value > b
        # ret, thresh = cv2.threshold(self.edges, 0, 50, cv2.THRESH_BINARY) # THRESH_BINARY: transform into black/white depending on low/high value
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        return contours

    ##  get center point using moment of edge
    def getXY(self, contours):
        radius = []
        n_contours = len(contours)
        cX = []
        cY = []
        for i in range(n_contours):
            c = contours[i]
            self.perimeters += [cv2.arcLength(c, True)]
            self.areas += [cv2.contourArea(c)]
            if (self.perimeters[-1] <= 6) | (self.areas[-1] <= 2) | (len(c) < 4):
                continue
            radius += [self.areas[-1]/self.perimeters[-1]]
            M = cv2.moments(c)
            #(1 < radius[-1] < 5) & (M['m10'] != 0) & (M['m01'] != 0)
            if (M['m00'] != 0):            
                self.radius_save += [radius[-1]]
                cX +=  [(M['m10'] / M['m00'])+32]
                cY +=  [(M['m01'] / M['m00'])+18]   
        return cX, cY

    ## remove beads are too close, choose two image, refer to smaller bead#
    def removeXY(self, cX, cY, criteria): 
        cX1 = np.array(cX) # len of cXr1 is smaller, as ref
        cY1 = np.array(cY)
        i_dele = np.empty(0).astype(int)
        for i in range(len(cX1)):
            dx = cX1 - cX1[i]
            dy = cY1 - cY1[i]
            dr = np.sqrt(dx**2 + dy**2)
            # i_dr = np.argsort(dr)
            # i_XYr_self = i_dr[0]
            if any(dr[dr != 0] <= criteria):
                i_dele = np.append(i_dele, int(i))                
        cX = np.delete(cX1, i_dele)
        cY = np.delete(cY1, i_dele)
        self.radius_save = np.delete(self.radius_save, i_dele)
        return cX, cY
        
    ##  get intensity of AOI
    def getintensity(self, image, cX, cY, aoi_size = 20): # i: bead number: 1,2,3,...,N
        half_size = int(aoi_size/2)    
        intensity = []
        for i in range(len(cX)):
            horizontal = int(cY[i]) # width
            vertical = int(cX[i])   # height
            intensity += [np.mean(image[horizontal-half_size:(horizontal+half_size), vertical-half_size:(vertical+half_size)])] # [x,y] = [width, height]
        intensity = np.array(intensity)
        return intensity
        
    ##  remove low intensity aoi
    def removeblack(self, cX, cY, intensity, blacklevel = 150):
        i_dele = np.empty(0).astype(int)
        for i in range(len(cX)):
            if intensity[i] < blacklevel:
                i_dele = np.append(i_dele, int(i))
        cX = np.delete(cX, i_dele)
        cY = np.delete(cY, i_dele)
        self.radius_save = np.delete(self.radius_save, i_dele)
        intensity = np.delete(intensity, i_dele)
        return cX, cY
        
    ##  plot X,Y AOI in given image
    def drawAOI(self, image, cX, cY, aoi_size = 20):
        n = len(cX)
        for i in range(n):
            cv2.circle(image, (int(cX[i]), int(cY[i])), aoi_size, (255, 255, 255), 1)
            cv2.putText(image, str(i+1), (int(cX[i]+10), int(cY[i]+10))
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        return image
    
    ##  show and save gray image
    def show_grayimage(self, image, save = True):
        plt.figure()
        plt.imshow(image, cmap='gray', vmin=0, vmax=255)
        if save == True:
            cv2.imwrite(os.path.join(self.path_folder, 'output.png'), image)

###############################################################################           
    ### methods for tracking beads
    ## get image-cut of certain AOI
    def getAOI(self, image, row, col, aoi_size=20):
        row = int(row) # cY, height
        col = int(col)   # cX, width
        size_half = int(aoi_size/2)
        image_cut = image[row-size_half:(row+size_half), col-size_half:(col+size_half)]
        intensity = np.sum(image_cut)
        return image_cut, intensity
    
    ## get sum of squared residuals
    def get_residuals(self, fn, x, y, image, popt):
        residuals = image.ravel() - fn((x, y), *popt)
        ss_res = np.sum(residuals**2)
        return ss_res
        
    ## get bounds for curve_fit
    def get_bounds(self, aoi_size=20):
        #(amplitude, sigma_x, sigma_y, xo, yo, theta_deg, offset)
        bounds=((0, 0, 0, 0, 0, 0, 0), (255, aoi_size, aoi_size, aoi_size-1, aoi_size-1, 90, 255))
        return bounds
    
    ## get parameters for trackbead fitting
    def preparefit_info(self, read_mode, frame_setread_num, frame_total, aoi_size = 20):
        bead_number = self.bead_number
        x_fit = np.array([[i for i in range(aoi_size)] for j in range(aoi_size)]).astype(float)
        y_fit = np.array([[j for i in range(aoi_size)] for j in range(aoi_size)]).astype(float)
        initial_guess = [40,3,3,11,11,0,10]
        parameters = [initial_guess] * bead_number
        if read_mode == 0:
            N = frame_setread_num
        else:
            N = frame_total
        return x_fit, y_fit, initial_guess, parameters, N
    
    def update_p0(self, p0_i, p0_f, i): #p0 is n by m matrix, n is bead number and m is 7, i=1,2,3,...
        p0 = (p0_i * i + p0_f)/(i+1)
        return p0
       
    ##  tracking bead position in a image, get all parameters and frame number
    def trackbead(self, image, cX, cY, aoi_size, frame, p0):
        para_fit = []
        bounds = self.get_bounds(aoi_size)
        x = self.x_fit
        y = self.y_fit
        parameters = self.parameters
        for j in range(self.bead_number):
            image_tofit, intensity = self.getAOI(image, cY[j], cX[j], aoi_size)
            ## enhance contrast
            # image_bead = image[horizontal-10:(horizontal+10), vertical-10:(vertical+10)] # [x,y] = [width, height]
            # image_bead_bur = cv2.GaussianBlur(image_bead, (5, 5),2,2)
            ## increase contrast
            # enh_con = ImageEnhance.Contrast(Image.fromarray(image_bead_bur))
            # contrast = 10
            # image_contrasted = enh_con.enhance(contrast)
            # image_tofit = np.array(image_contrasted)
            try:
                popt, pcov = opt.curve_fit(twoD_Gaussian, [x, y],image_tofit.ravel(), p0[j], bounds=bounds)
                ss_res = self.get_residuals(twoD_Gaussian, x, y, image_tofit, popt)
                # popt: optimized parameters, pcov: covariance of popt, diagonal terms are variance of parameters
                # data_fitted = twoD_Gaussian((x, y), *popt)
                intensity_integral = 2*math.pi*popt[0]*popt[1]*popt[2]
                para_fit += [
                    [frame] + [j+1] + list(popt) + 
                    [intensity] + [intensity_integral] + [ss_res]
                             ]
                parameters[j] = list(popt)
                
            except RuntimeError:
                # popt, pcov = opt.curve_fit(twoD_Gaussian, [x, y],image_tofit.ravel(), initial_guess)
                para_fit += [[frame] + [j+1]+[0,0,0,0,0,0,0,0,0,0]]
                parameters[j] = self.initial_guess # initial guess for all beads
            except:
                para_fit += [[frame] + [j+1]+[0,0,0,0,0,0,0,0,0,0]]
                parameters[j] = self.initial_guess
        return para_fit, np.array(parameters)
    
###############################################################################
    ### methods for image reading    
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

###############################################################################        
    ### methods for getting header information
    def getheader(self):
        if platform == 'win32':
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
                PixelDepth.value, timeOf1stFrameSecSince1104.value]
            # [frames, height, width, pixeldepth, avg fps]
            return self.header
        else: # is linux
            df = pd.read_csv(self.path_header_txt, sep='\t', header=None)
            header_names = df[0].to_numpy()
            header_values = df[1].to_numpy()
            self.header = [int(header_values[0]), int(header_values[2]), int(header_values[1]), int(header_values[4]), header_values[3]]
            [self.frames_acquired, self.height, self.width, self.pixeldepth, self.avg_fps] = self.header
            # [frames, height, width, pixeldepth, avg fps]
            return self.header

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
        info = [self.frame_per_file, self.path_data, self.data_type, self.size_a_image]
        return info

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
        # self.offset = offset
        # self.fileNumber = fileNumber
        return offset, fileNumber   
###############################################################################
### Use for data saving and data reshaping
class DataToSave:
    # data: np.array, path_folder: string path
    def __init__(self, data, path_folde, time_axis):
        
        self.columns = self.get_saved_name()
        self.df = pd.DataFrame(data=data, columns=self.columns)
        self.path_folder = path_folder
        self.time_axis = time_axis
        self.filename_time = self.get_date()
        self.bead_number = int(max(self.df['aoi']))
        self.frame_acquired = int(len(self.df['x'])/self.bead_number)
        
    ##  save data to csv
    def save_original_data(self):
        path_folder = self.path_folder
        filename_time = self.filename_time
        self.df.to_csv(os.path.join(path_folder, filename_time + '-fitresults.csv'), index=False)    
        
    ##  get sx*sy and sx/sy ,and write reshape data to excel
    def save_reshape_data(self):
        df = self.df
        path_folder = self.path_folder
        filename_time = self.filename_time
        bead_number = self.bead_number
        frame_acquired = self.frame_acquired
        df.insert(5, 'sx_sy', df['sx']*df['sy'])
        df.insert(6, 'sx_over_sy', df['sx']/df['sy'])
        df_time = pd.DataFrame(data=self.time_axis)
        writer = pd.ExcelWriter(os.path.join(path_folder, f'{filename_time}-fitresults_reshape.xlsx'))
        for i, element in enumerate(df.columns):
            if i > 1:
                name = self.assign_name(element, bead_number)
                data = self.get_attrs(df[element], bead_number, frame_acquired)
                df_reshape = pd.DataFrame(data=data, columns=name)
                df_reshape.insert(0, 'time', df_time)
                df_reshape.to_excel(writer, sheet_name=element)
        writer.save()
        return df_reshape, bead_number, frame_acquired
        
    ##  columns to be saved
    def get_saved_name(self):
        columns = ['frame', 'aoi', 'amplitude', 'sx', 'sy', 'x', 'y', 'theta_deg', 'offset', 'intensity', 'intensity_integral', 'ss_res']
        return columns
    
    ##  getting date
    def get_date(self):
        filename_time = datetime.datetime.today().strftime('%Y-%m-%d') # yy-mm-dd
        return filename_time

    ##  assign anme for each column to be saved
    def assign_name(self, name, bead_number):
        name = [f'{name}_{i+1}' for i in range(bead_number)]
        return name
    
    ##  input 1D array data, output: (row, column) = (frame, bead)
    def get_attrs(self, data_col, bead_number, frame_acquired):
        data_col = np.array(data_col)
        data_col_reshape = np.reshape(data_col, (frame_acquired, bead_number))
        return data_col_reshape     


###  2-D Gaussian function with rotation angle
def twoD_Gaussian(xy, amplitude, sigma_x, sigma_y, xo, yo, theta_deg, offset):
    xo = float(xo)
    yo = float(yo)
    theta = theta_deg/360*(2*math.pi) # in rad
    x = xy[0]
    y = xy[1]    
    a = (np.cos(theta)**2)/(2*sigma_x**2) + (np.sin(theta)**2)/(2*sigma_y**2)
    b = -(np.sin(2*theta))/(4*sigma_x**2) + (np.sin(2*theta))/(4*sigma_y**2)
    c = (np.sin(theta)**2)/(2*sigma_x**2) + (np.cos(theta)**2)/(2*sigma_y**2)
    g = offset + amplitude*np.exp( - (a*((x-xo)**2) + 2*b*(x-xo)*(y-yo) 
                                    + c*((y-yo)**2)) )
    return g.ravel()  




# ###  tracking bead position in a image, get center and std of X,Y using Gaussian fit
# def trackbead(image):  
#     xc = []
#     yc = []
#     sx = []
#     sy = []
#     para_fit = []
#     bounds = get_bounds(aoi_size)
#     for j in range(bead_number):
#         image_tofit, intensity = getAOI(image, aoi[1][j], aoi[0][j])
#         ## enhance contrast
#         # image_bead = image[horizontal-10:(horizontal+10), vertical-10:(vertical+10)] # [x,y] = [width, height]
#         # image_bead_bur = cv2.GaussianBlur(image_bead, (5, 5),2,2)
#         ## increase contrast
#         # enh_con = ImageEnhance.Contrast(Image.fromarray(image_bead_bur))
#         # contrast = 10
#         # image_contrasted = enh_con.enhance(contrast)
#         # image_tofit = np.array(image_contrasted)
#         popt=[]
#         pcov=[]
#         try:
#             popt, pcov = opt.curve_fit(twoD_Gaussian, [x, y],image_tofit.ravel(), parameters[j], bounds=bounds)
#             ss_res = get_residuals(twoD_Gaussian, x, y, image_tofit, popt)
#             # popt: optimized parameters, pcov: covariance of popt, diagonal terms are variance of parameters
#             # data_fitted = twoD_Gaussian((x, y), *popt)
#             xc += [popt[1]] # x position of each frame
#             yc += [popt[2]] #
#             sx += [popt[3]]
#             sy += [popt[4]]
#             para_fit += [[j+1] + list(popt) + [intensity] + [ss_res]]
#             parameters[j] = list(popt)
            
#         except RuntimeError:
#             # popt, pcov = opt.curve_fit(twoD_Gaussian, [x, y],image_tofit.ravel(), initial_guess)
#             xc += [0] # x position of each frame
#             yc += [0] #  
#             sx += [0]
#             sy += [0]
#             para_fit += [[j+1]+[0,0,0,0,0,0,0,0,0]]
#             parameters[j] = [initial_guess] # initial guess for all beads
#         except:
#             para_fit += [[j+1]+[0,0,0,0,0,0,0,0,0]]

#     # return [xc, yc, sx, sy, para_fit]
#     return para_fit


# ### get parameters for each fitting loop
# def getloop_info(frame_start, frame_total, size_tofit):

#     frame_i = []
#     N = []
#     i_run = np.ceil(frame_total/size_tofit).astype(int)
#     for i in range(i_run):
#         frame_i += [ min((i)*size_tofit, frame_total) ]
#         if frame_total>(i+1)*size_tofit:
#             N += [size_tofit]
#         else:
#             N += [frame_total-(i)*size_tofit]
#     return frame_i, N # frame_i: frame_start for each loop, N: size to fit for each loop

# ### get csv writting info
# def getcsvinfo(bead_number):
#     bead_namex = []
#     bead_namey = []
#     bead_namesx = []
#     bead_namesy = []
#     bead_nameI = []
#     ##   create csv file to store xy position data
#     for i in range(bead_number):
#         bead_namex += ['beadx '+str(i+1)]
#         bead_namey += ['beady '+str(i+1)]
#         bead_namesx += ['stdx '+str(i+1)]
#         bead_namesy += ['stdy '+str(i+1)]
#         bead_nameI += ['intensity '+str(i+1)]

#     bead_namexy = bead_namex + bead_namey + bead_namesx + bead_namesy + bead_nameI
#     return bead_namexy

# ### information for multi-threading
# def print_start(frame_i, frame_f):
#     print('start analyzing ' + 'frame: ' + str(frame_i) + ' to ' + str(frame_f) )

# def print_finish(frame_i, frame_f, frame_total):
#     print('finish analyzing ' + 'frame: ' + str(frame_i) + ' to ' + str(frame_f) + ', progress: ' +
#                           str(int(100*frame_f/frame_total)) + '%')


# ### main to tracking each beads over each frame
# def fit_all_frame(Gdata, frame_start, N, size_tofit):
#     frame_i, frame_tofit = getloop_info(frame_start, N, size_tofit)
#     bead_namexy = getcsvinfo(len(Gdata.cX))
#     result = []
#     ##  open .csv to be saved
#     with open(file_folder+'/'+filename_time+'-xy and sigma xy.csv', 'w', newline='') as csvfile:
#         writer = csv.writer(csvfile)
#         writer.writerow(bead_namexy)
#         for i, n in zip(frame_i, frame_tofit):
#             image_eachframe = Gdata.readGlimpseN(i, n)
#             print_start(i, i+n)
#             # r = fit_mode(image_eachframe, fit_mode = fit_mode)
#             # result.append(r)
#             with mp.Pool(mp.cpu_count()-4) as pool:
#                 #freeze_support()
#                 r = pool.map(trackbead, image_eachframe)
#                 #pool.close()
#                 #pool.join()
#                 result += r
#                 if len(result) == N:
#                     data = np.array(result)
#                     for k in range(len(result)): # number of frame
#                     ##  save x,y,sx,sy 
#                         writer.writerow(list(data[k][:,1]) + list(data[k][:,2]) + list(data[k][:,3]) + list(data[k][:,4]) + list(data[k][:,7]))
#                     print_finish(i, i+n, N)
#                     print('saving...')
#                 else:
#                     print_finish(i, i+n, N)
#     return result, r


if __name__ == "__main__":
    t1 = time.time()
    
    Glimpse_data = BinaryImage(path_folder, criteria_dist = criteria_dist, aoi_size = aoi_size,
                        frame_read_forcenter = frame_read_forcenter, N_loc = N_loc,
                        contrast = contrast, low = low, high = high, blacklevel = blacklevel)
    image, cX, cY = Glimpse_data.Localize() # localize beads
    result_saved, time_axis = Glimpse_data.Track_All_Frames(read_mode, frame_setread_num)
    Save_df = DataToSave(result_saved, path_folder, time_axis)
    Save_df.save_original_data() # save -fitresult.csv
    Save_df.save_reshape_data()  # save -fitresult_reshape.xlsx

    time_spent = time.time() - t1
    print('spent ' + str(time_spent) + ' s')





