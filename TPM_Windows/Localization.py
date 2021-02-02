# -*- coding: utf-8 -*-
"""
Created on Fri Oct 23 13:30:44 2020
version: 20201113
use openCV to localize bead position
and save to .aoi
version: 20201210
two classes, for glimpse information(data_folder) and data manipulatin(Gimage)
@author: YCC

"""

### setting parameters
path_folder = r'C:\Users\OT-hwLi-lab\Desktop\YCC\20210127\qdot655\3281bp\3\4-200ms-110uM_BME'

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


### getting parameters
today = datetime.datetime.now()
filename_time = str(today.year)+str(today.month)+str(today.day)

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
            cv2.circle(self.image, (int(self.cX[i]), int(self.cY[i])), self.AOI_size, (255, 255, 255), 1)
            cv2.putText(self.image, str(i+1), (int(self.cX[i]+10), int(self.cY[i]+10))
                        , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
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
    
    Gdata.contrast(contrast = contrast)
    Gdata.getContour(low, high)
    Gdata.getXY()
    Gdata.removeXY()    
    Gdata.getintensity(aoisize = 2)
    Gdata.removeblack(blacklevel = blacklevel)
    Gdata.drawAOI()
    print('finish centering')
    return Gdata, folder

# ##  Read glimpse parameters
# def getparameters():
#     folder = data_folder(path_folder)
#     folder.getheader()   
#     folder.getdatainfo()
#     return folder.header, folder.size_a_image, folder.path_data, folder.frame_per_file, folder.frame_total



# ### get parameters
# header, size_a_image, file_name, frame_per_file, frame_total = getparameters()

### main to get localization information
if __name__ == '__main__':
    Gdata, folder = localize(path_folder)
    image_localization = Gdata.image
    cX = Gdata.cX
    cY = Gdata.cY
    plt.figure()
    plt.imshow(image_localization, cmap='gray', vmin=0, vmax=255)
    # string_save=['']
    # for i in range(len(cX)):
    #     string_save += [str(i+1) +' '+ str(int(cX[i]-aoi_size/2)) +' '+ 
    #                     str(int(cY[i]-aoi_size/2)) +' '+ str(aoi_size) +' '+
    #                     str(aoi_size) + '\n']
    
    # with open(filename_time+"-localization.aoi", "w") as aoi:
    #     aoi.writelines(string_save)



# cv2.matchShapes(contours[0],contours[1],1, 0.0)




