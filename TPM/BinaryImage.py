### import used modules first
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

### define a class for all glimpse data
class BinaryImage:
    def __init__(self, path_folder, read_mode=1, frame_setread_num=20, frame_start=0,
                 criteria_dist=20, aoi_size=20, frame_read_forcenter=0,
                  N_loc=40, contrast=10, low=40, high=120,
                 blacklevel=30, whitelevel=200):
        self.random_string = self.__gen_random_code(3)
        self.path_folder = os.path.abspath(path_folder)
        self.path_header = os.path.abspath(os.path.join(path_folder, 'header.glimpse'))
        self.path_header_utf8 = self.path_header.encode('utf8')
        self.path_header_txt = os.path.abspath(os.path.join(path_folder, 'header.txt'))
        self.path_data = self.__get_path_data()
        [self.frames_acquired, self.height, self.width, self.pixeldepth, self.med_fps] = self.getheader()
        self.data_type, self.size_a_image, self.frame_per_file = self.__getdatainfo()
        self.read_mode = read_mode
        self.frame_setread_num = frame_setread_num
        self.criteria_dist = criteria_dist
        self.aoi_size = aoi_size
        self.frame_read_forcenter = frame_read_forcenter
        self.frame_start = frame_start
        self.N_loc = N_loc
        self.contrast = contrast
        self.low = low
        self.high = high
        self.blacklevel = blacklevel
        self.whitelevel = whitelevel
        self.offset, self.fileNumber = self.__getoffset()
        self.cut_image_width = 30
        self.readN = self.__readGlimpseN(frame_read_forcenter, N_loc)  # N image from i
        self.image = self.__stackimageN(self.readN)  # image used to be localized
        self.x_fit = np.array([[i for i in range(aoi_size)] for j in range(aoi_size)]).astype(float)
        self.y_fit = np.array([[j for i in range(aoi_size)] for j in range(aoi_size)]).astype(float)
        self.background = np.mean(self.image)
        self.initial_guess = [50., 2., 2., aoi_size/2, aoi_size/2, 0., self.background]
        self.image_cut = []


    ###########################################################################
    ##  main for localization
    def Localize(self, put_text=True):
        print('start centering')
        image = self.image
        image = self.__enhance_contrast(image, self.contrast)
        contours = self.getContour(image, self.low, self.high)
        cX, cY = self.getXY(contours)

        ##  need to sort according to X first and select
        for i in range(2):
            cX, cY = self.__sortXY(cX, cY)
            cX, cY = self.select_XY(cX, cY, self.criteria_dist)
            cX, cY, amplitude = self.get_accurate_xy(image, cX, cY)
            cX, cY, amplitude = self.removeblack(cX, cY, amplitude, self.blacklevel)

        self.bead_number = len(cX)
        image = self.__drawAOI(image, cX, cY, self.aoi_size, put_text=put_text)
        self.__show_grayimage(image, save=True)
        self.cX = cX
        self.cY = cY
        self.image = image
        print('finish centering')
        bead_radius = self.radius_save.reshape((-1,1))
        random_string = self.random_string
        return bead_radius, random_string

    ##  main for tracking all frames and all beads(cX, cY)
    def Track_All_Frames(self, selected_aoi=None, IC=False):

        frames_acquired = self.frames_acquired
        frame_start = self.frame_start
        aoi_size = self.aoi_size
        read_mode = self.read_mode
        frame_setread_num = self.frame_setread_num
        frame_read_forcenter = self.frame_read_forcenter
        initial_guess, initial_guess_beads, N = self.__preparefit_info(read_mode, frame_setread_num, frames_acquired)
        if selected_aoi == None:
            cX = self.cX
            cY = self.cY
        else:
            cX = np.array(self.cX[selected_aoi], ndmin=1)
            cY = np.array(self.cY[selected_aoi], ndmin=1)
            initial_guess_beads = np.array(initial_guess_beads[selected_aoi], ndmin=2)

        p0_1 = initial_guess_beads  # initialize fitting parameters for each bead
        tracking_results_list = []

        for i in range(N):
            image = self.__readGlimpse1(frame_start+i)
            data, p0_2 = self.trackbead(image, cX, cY, aoi_size, frame=i, initial_guess_beads=p0_1,IC=IC)
            # p0_1 = self.__update_p0(p0_1, p0_2, i)  # update fitting initial guess
            tracking_results_list += data
            print(f'frame {i}')
        self.N = N
        self.initial_guess_beads = p0_1
        tracking_results = np.array(tracking_results_list)
        self.tracking_results = tracking_results
        self.aoi = [cX, cY]
        return tracking_results

    ##  main for getting fit-video of an aoi
    def Get_fitting_video_offline(self, selected_aoi, frame_i, N):
        tracking_results = self.tracking_results
        # cX = self.cX
        # cY = self.cY
        cX, cY = self.aoi
        x = self.x_fit
        y = self.y_fit
        n_fit = len(x)
        aoi_size = self.aoi_size
        path_folder = self.path_folder
        tracking_results_select = self.get_aoi_from_tracking_results(tracking_results, selected_aoi)
        imageN = self.__readGlimpseN(frame_i, N=N)
        fourcc = cv2.VideoWriter_fourcc(*'H264')
        output_movie = cv2.VideoWriter(os.path.abspath(path_folder) + f'/{self.random_string}-fitting2.mp4', fourcc, 5.0, (1200, 800))
        i=0
        for image, tracking_result_select in zip(imageN, tracking_results_select):
            image_aoi, intensity = self.__getAOI(image, cY[selected_aoi], cX[selected_aoi], aoi_size)
            para_fit = tracking_result_select[2:9]
            data_fitted = twoD_Gaussian((x, y), *para_fit)
            fig, ax = plt.subplots(1, 1)
            # ax.imshow(image_aoi, cmap=plt.cm.gray, origin='lower',
            #           extent=(x.min(), x.max(), y.min(), y.max()))
            ax.imshow(image_aoi)

            ax.contour(x, y, data_fitted.reshape(n_fit, n_fit), 5, colors='r')
            # ax.contour(x, y, data_fitted.reshape(20, 20), 5)
            plot_img_np = self.get_img_from_fig(fig)
            plot_img_np = cv2.resize(plot_img_np, (1200, 800))
            output_movie.write(plot_img_np)
            plt.close()
            print(f'storing frame {i}')
            i+= 1
        self.image_aoi = image_aoi
        self.ax = ax
        output_movie.release()

    ###############################################################################
    ##  get accurate position using Gaussian fit
    def get_accurate_xy(self, image, cX, cY):
        aoi_size = self.aoi_size
        initial_guess_beads = np.array([self.initial_guess] * len(cX))
        data, popt_beads = self.trackbead(image, cX, cY, aoi_size, frame=0, initial_guess_beads=initial_guess_beads)
        amplitude = popt_beads[:, 0]
        x = popt_beads[:, 3]
        y = popt_beads[:, 4]
        self.dx_localization = x - aoi_size/2
        self.dy_localization = y - aoi_size/2
        self.initial_guess_beads = initial_guess_beads
        self.amplitude = amplitude
        cX = cX + self.dx_localization
        cY = cY + self.dy_localization
        return cX, cY, amplitude

    ##  tracking position of all beads in a image, get all parameters and frame number
    def trackbead(self, image, cX, cY, aoi_size, frame, initial_guess_beads, IC=False):
        bead_number = len(cX)
        data = []
        bounds = self.__get_bounds(aoi_size)
        x = self.x_fit
        y = self.y_fit
        # initial_guess_beads = self.initial_guess_beads
        initial_guess = self.initial_guess
        for j in range(bead_number):
            image_tofit, intensity = self.__getAOI(image, cY[j], cX[j], aoi_size)
            initial_guess = self.__get_guess(image_tofit)
            # x_guess = np.argmax(image_tofit)%aoi_size
            # y_guess = np.argmax(image_tofit)//aoi_size
            # initial_guess[3:5] = [x_guess, y_guess]
            if IC==True:
                contrast = 8
                image_tofit = ImageEnhance.Contrast(Image.fromarray(image_tofit.astype('uint8'))).enhance(contrast)
                image_tofit = np.array(image_tofit)
            # if intensity < 3500:
            #     contrast = 2
            #     image_tofit = ImageEnhance.Contrast(Image.fromarray(image_tofit.astype('uint8'))).enhance(contrast)
            #     image_tofit = np.array(image_tofit)
            ## enhance contrast
            # image_bead = image[horizontal-10:(horizontal+10), vertical-10:(vertical+10)] # [x,y] = [width, height]
            # image_bead_bur = cv2.GaussianBlur(image_bead, (5, 5),2,2)
            ## increase contrast
            # enh_con = ImageEnhance.Contrast(Image.fromarray(image_bead_bur))
            # contrast = 10
            # image_contrasted = enh_con.enhance(contrast)
            # image_tofit = np.array(image_contrasted)
            try:
                # popt, pcov = opt.curve_fit(twoD_Gaussian, [x, y], image_tofit.ravel(), initial_guess_beads[j, :],
                #                            bounds=bounds)
                popt, pcov = opt.curve_fit(twoD_Gaussian, [x, y], image_tofit.ravel(), initial_guess,
                                           bounds=bounds, method='trf')
                ss_res = self.__get_residuals(twoD_Gaussian, x, y, image_tofit, popt)
                # popt: optimized parameters, pcov: covariance of popt, diagonal terms are variance of parameters
                # data_fitted = twoD_Gaussian((x, y), *popt)
                intensity_integral = 2 * math.pi * popt[0] * popt[1] * popt[2]
                data += [
                    [frame] + [j] + list(popt) +
                    [intensity] + [intensity_integral] + [ss_res]
                ]
                # initial_guess_beads[j] = list(popt)
                initial_guess_beads[j, :] = popt
            except RuntimeError:
                # popt, pcov = opt.curve_fit(twoD_Gaussian, [x, y],image_tofit.ravel(), initial_guess)
                data += [[frame] + [j] + [0.] * 10]
                initial_guess_beads[j, :] = np.array(initial_guess)  # initial guess for all beads
            except:
                data += [[frame] + [j] + [0.] * 10]
                initial_guess_beads[j, :] = np.array(initial_guess)
        popt_beads = np.array(initial_guess_beads)
        return data, popt_beads

    ### methods for localization


    ##  get egde using frame in file,f
    def getContour(self, image, low=30, high=90):
        cut = self.cut_image_width
        ##  get edges using openCV
        image_cut = np.uint8(image[0 + cut:self.height - cut, 0 + cut:self.width - cut])
        edges = cv2.Canny(image_cut, low, high)  # cv2.Canny(image, a, b), reject value < a and detect value > b
        # ret, thresh = cv2.threshold(self.edges, 0, 50, cv2.THRESH_BINARY) # THRESH_BINARY: transform into black/white depending on low/high value
        contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        self.edges = edges
        self.contours = contours
        return contours

    ##  get center point using moment of edge
    def getXY(self, contours):
        cut = self.cut_image_width
        radius = []
        radius_save = []
        n_contours = len(contours)
        cX = []
        cY = []
        saved_contours = []
        perimeters = []
        areas = []
        for i in range(n_contours):
            c = contours[i]
            perimeters += [cv2.arcLength(c, True)]
            areas += [cv2.contourArea(c)]
            # if (perimeters[-1] <= 0) | (areas[-1] <= 0) | (len(c) < 2):
            if (perimeters[-1] == 0):
                continue  ## ingore code below
            radius += [2 * areas[-1] / perimeters[-1]]  ## r^2/2r = r/2
            M = cv2.moments(c)
            # if (M['m00'] != 0) & (radius[-1] > 1):
            if (M['m00'] != 0):
                radius_save += [radius[-1]]
                saved_contours += [c]
                cX += [(M['m10'] / M['m00']) + cut]
                cY += [(M['m01'] / M['m00']) + cut]
        self.perimeters = perimeters
        self.areas = areas
        self.saved_contours = np.array(saved_contours)
        cX = np.array(cX)
        cY = np.array(cY)
        radius_save = np.array(radius_save)
        self.radius_save = radius_save
        return cX, cY

    ##  core for selecting points which are not too close
    def select_XY(self, cX, cY, criteria):
        cX1 = np.array(cX)
        cY1 = np.array(cY)
        n = len(cX1)
        cX_selected = np.empty(0)
        cY_selected = np.empty(0)
        r_selected = np.empty(0)
        index = []
        for i in range(n):
            x2 = cX1[i]
            y2 = cY1[i]
            r = np.sqrt(x2**2 + y2**2)
            c1 = abs(x2-cX_selected) >= criteria
            c2 = abs(y2-cY_selected) >= criteria
            c3 = abs(r-r_selected) >= criteria
            c = np.array([c1 or c2 or c3 for c1,c2,c3 in zip(c1,c2,c3)]) ## get boolean array for outside of criteria distance
            if all(c) or (i==0): ## collecting centers, every point should qualify
                cX_selected = np.append(cX_selected, x2)
                cY_selected = np.append(cY_selected, y2)
                r = np.sqrt(x2**2 + y2**2)
                r_selected = np.append(r_selected, r)
                index += [i]
        self.radius_save = self.radius_save[index]
        self.saved_contours = self.saved_contours[index]
        return cX_selected, cY_selected

    ## remove beads are too close, choose two image, refer to smaller bead#
    def removeXY(self, cX, cY, criteria):
        cX1 = np.array(cX)  # len of cXr1 is smaller, as ref
        cY1 = np.array(cY)
        i_dele = np.empty(0).astype(int)
        for i in range(len(cX1)):
            dx = cX1 - cX1[i]
            dy = cY1 - cY1[i]
            dr = np.sqrt(dx ** 2 + dy ** 2)
            if any(dr[dr != 0] <= criteria):
                i_dele = np.append(i_dele, int(i))
        cX = np.delete(cX1, i_dele)
        cY = np.delete(cY1, i_dele)
        self.radius_save = np.delete(self.radius_save, i_dele)
        self.saved_contours = np.delete(self.saved_contours, i_dele)
        return cX, cY

    ##  get avg intensity of all AOI(20 * 20 pixel)
    def getintensity(self, image, cX, cY, aoi_size=20):  # i: bead number: 1,2,3,...,N
        half_size = int(aoi_size/2)
        intensity = []
        for i in range(len(cX)):
            horizontal = int(cY[i])  # width
            vertical = int(cX[i])  # height
            intensity += [np.mean(image[horizontal - half_size:(horizontal + half_size),
                                  vertical - half_size:(vertical + half_size)])]  # [x,y] = [width, height]
        intensity = np.array(intensity)
        return intensity

    ##  remove low intensity aoi
    def removeblack(self, cX, cY, amplitude, blacklevel=50):
        amplitude = np.array(amplitude)
        index = amplitude >= blacklevel ## get amplitude >= blacklevel
        cX = cX[index]
        cY = cY[index]
        amplitude = amplitude[index]
        self.initial_guess_beads = self.initial_guess_beads[index]
        self.amplitude = amplitude
        self.radius_save = self.radius_save[index]
        self.saved_contours = self.saved_contours[index]
        return cX, cY, amplitude

    ##  remove high intensity aoi
    def removewhite(self, cX, cY, amplitude, whitelevel=150):
        amplitude = np.array(amplitude)
        index = amplitude <= whitelevel ## get amplitude >= blacklevel
        cX = cX[index]
        cY = cY[index]
        amplitude = amplitude[index]
        self.amplitude = amplitude
        self.radius_save = self.radius_save[index]
        self.saved_contours = self.saved_contours[index]
        return cX, cY, amplitude

    ##  sort bead number using distance between y-axis(x = 0), distance from left-up corner
    def __sortXY(self, cX, cY):
        n = len(cX)
        R = np.sqrt(cX**2 + cY**2)
        index = np.argsort(R)
        cX = cX[index]
        cY = cY[index]
        self.radius_save = np.reshape(self.radius_save[index], (n, 1))
        self.saved_contours = self.saved_contours[index]
        return cX, cY

    ##  stack multiple images
    def __stackimageN(self, imageN):
        return np.mean((imageN.T), 2).T.astype('uint8')

    ##  enhance contrast
    def __enhance_contrast(self, image, contrast=10):
        enh_con = ImageEnhance.Contrast(Image.fromarray(image))
        image_contrasted = enh_con.enhance(contrast)
        image = np.array(image_contrasted)
        return image

    ##  plot X,Y AOI in given image
    def __drawAOI(self, image, cX, cY, aoi_size=20, put_text=True):
        n = len(cX)
        for i in range(n):
            cv2.circle(image, (int(cX[i]), int(cY[i])), aoi_size, (255, 255, 255), 1)
            if put_text == True:
                cv2.putText(image, str(i), (int(cX[i] + aoi_size/2), int(cY[i] + aoi_size/2))
                            , cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 1, cv2.LINE_AA)
        return image

    ##  show and save gray image
    def __show_grayimage(self, image, save=True):
        random_string = self.random_string
        plt.ion()
        fig, ax = plt.subplots()
        # plt.figure()
        ax.imshow(image, cmap='gray', vmin=0, vmax=255)
        pylab.show()
        if save == True:
            cv2.imwrite(os.path.join(self.path_folder, random_string + '-output.png'), image)

    ##  add 2n-word random texts(n-word number and n-word letter)
    def __gen_random_code(self, n):
        digits = "".join([random.choice(string.digits) for i in range(n)])
        chars = "".join([random.choice(string.ascii_letters) for i in range(n)])
        return digits + chars

    ###############################################################################
    ### method for making video of certain aoi, tracking_results: list array
    ##  get tracking result for assigned aoi
    def get_aoi_from_tracking_results(self, tracking_results, selected_aoi):
        # frame_i = int(min(tracking_results[:,0]))
        frame_acquired = int(max(tracking_results[:, 0]) + 1)
        bead_number = int(max(tracking_results[:, 1]) + 1)
        tracking_results_list = list(tracking_results)
        indices_select = [x * bead_number + selected_aoi for x in range(frame_acquired)]
        tracking_results_select = []
        for i in indices_select:
            tracking_results_select += [tracking_results_list[i]]
        return np.array(tracking_results_select)

    ## define a function which returns an image as numpy array from figure
    def get_img_from_fig(self, fig, dpi=200):
        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=dpi)
        buf.seek(0)
        img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
        buf.close()
        img = cv2.imdecode(img_arr, 1)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # cv2.COLOR_BGR2RGB
        return img

    ###############################################################################
    ### methods for tracking beads
    ## get image-cut of certain AOI
    def __getAOI(self, image, row, col, aoi_size=20):
        row = int(row)  # cY, height
        col = int(col)  # cX, width
        size_half = int(aoi_size / 2)
        image_cut = image[row - size_half:(row + size_half), col - size_half:(col + size_half)]
        intensity = np.sum(image_cut)
        return image_cut, intensity

    ## get sum of squared residuals
    def __get_residuals(self, fn, x, y, image, popt):
        residuals = image.ravel() - fn((x, y), *popt)
        ss_res = np.sum(residuals ** 2)
        return ss_res

    ## get bounds for curve_fit
    def __get_bounds(self, aoi_size=20):
        ## (amplitude, sigma_x, sigma_y, xo, yo, theta_deg, offset)
        bounds = ((1, 0.5, 0.5, 0, 0, 0, 0), (255, aoi_size/2, aoi_size/2, aoi_size-1, aoi_size-1, 90, 255))
        self.bounds = bounds
        return bounds

    ## get parameters for trackbead fitting
    def __preparefit_info(self, read_mode, frame_setread_num, frame_total):
        initial_guess = self.initial_guess
        initial_guess_beads = self.initial_guess_beads
        if read_mode == 0:
            N = frame_setread_num
        else:
            N = frame_total
        return initial_guess, initial_guess_beads, N

    def __update_p0(self, p0_i, p0_f, i):  # p0 is n by m matrix, n is bead number and m is 7, i=0,1,2,3,...
        i += 1
        p0 = (p0_i * i + p0_f) / (i + 1)
        return p0

    def __get_guess(self, image_tofit):
        aoi_size = self.aoi_size
        amp_guess = np.max(image_tofit)
        x_guess = np.argmax(image_tofit) % aoi_size
        y_guess = np.argmax(image_tofit) // aoi_size
        background = self.background
        initial_guess = [amp_guess, 2.5, 2.5, x_guess, y_guess, 0, background]
        return initial_guess

    ###############################################################################
    ### methods for image reading
    ##  read one image at frame_i (0,1,2,...,N-1)
    def __readGlimpse1(self, frame_i=0):
        fileNumber = self.fileNumber[frame_i]
        offset = self.offset[frame_i]
        size_a_image = self.size_a_image
        data_type = self.data_type
        height = self.height
        width = self.width
        with open(self.path_data[fileNumber], 'rb') as f:
            f.seek(offset)
            data = f.read(size_a_image)
            decoded_data = struct.unpack('>' + str(size_a_image * 1) + data_type, data)
            read1 = np.reshape(decoded_data, (height, width))
            # self.image = self.read1
        return read1

    ##  read N image from frame_i (0,1,2,...,N-1)
    def __readGlimpseN(self, frame_i=0, N=50):
        fileNumber = self.fileNumber[frame_i: frame_i + N]
        offset_toread = [self.offset[x] for x in set(fileNumber)]
        path_toread = [self.path_data[x] for x in set(fileNumber)]
        frame_toread = [sum(fileNumber == x) for x in set(fileNumber)]
        decoded_data = []
        for path, frame, offset in zip(path_toread, frame_toread, offset_toread):
            with open(path, 'rb') as f:
                f.seek(offset)
                data = f.read(self.size_a_image * frame)
                decoded_data += struct.unpack('>' + str(self.size_a_image * frame) + self.data_type, data)
        readN = np.reshape(decoded_data, (N, self.height, self.width))
        return readN

    ###############################################################################
    ### methods for getting header information
    def getheader(self):
        if platform == 'win32':
            try:
                mydll = ctypes.windll.LoadLibrary('./GetHeader.dll')
            except:
                mydll = ctypes.windll.LoadLibrary('TPM/GetHeader.dll')
            GetHeader = mydll.ReadHeader  # function name is ReadHeader
            # assign variable first (from LabVIEW)
            # void ReadHeader(char String[], int32_t *offset, uint8_t *fileNumber,
            # uint32_t *PixelDepth, double *timeOf1stFrameSecSince1104 (med. fps (Hz)),uint32_t *Element0OfTTB,
            # int32_t *RegionHeight, int32_t *RegionWidth,
            # uint32_t *FramesAcquired)
            # ignore array datatype in header.glimpse
            GetHeader.argtypes = (ctypes.c_char_p, ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_uint),
                                  ctypes.POINTER(ctypes.c_uint), ctypes.POINTER(ctypes.c_double),
                                  ctypes.POINTER(ctypes.c_uint),
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
                      FramesAcquired)  # There are 8 variables.
            self.header = [FramesAcquired.value, RegionHeight.value, RegionWidth.value,
                           PixelDepth.value, timeOf1stFrameSecSince1104.value]
            ## header = [frames, height, width, pixeldepth, med fps]
            return self.header
        else:  # is linux or others
            df = pd.read_csv(self.path_header_txt, sep='\t', header=None)
            # header_columns = df[0].to_numpy()
            header_values = df[1].to_numpy()
            self.header = [int(header_values[0]), int(header_values[2]), int(header_values[1]), int(header_values[4]),
                           header_values[3]]
            [self.frames_acquired, self.height, self.width, self.pixeldepth, self.med_fps] = self.header
            # header = [frames, height, width, pixeldepth, med fps]
            return self.header

    ##  remove empty files and sort files by last modified time
    def __get_path_data(self):
        all_path = [os.path.abspath(x) for x in sorted(glob(os.path.join(self.path_folder, '*.glimpse'))) if
         x != self.path_header]
        ##  remove data that size is 0 byte
        all_path = [path for path in all_path if Path(path).stat().st_size != 0]
        all_modif_time = [os.path.getmtime(path) for path in all_path]
        all_path = [all_path[i] for i in np.argsort(all_modif_time)]
        return all_path


    def __getdatainfo(self):
        ### get file info.
        header = self.header
        path_data = self.path_data
        if header[3] == 0:  # 8 bit integer
            data_type = 'B'
            pixel_depth = 1
        else:
            data_type = 'h'
            pixel_depth = 2
        size_a_image = header[1] * header[2] * pixel_depth  # 8bit format default
        file_size = [Path(x).stat().st_size for x in path_data]
        frame_per_file = [int(x / size_a_image) for x in file_size]
        self.data_type, self.size_a_image, self.frame_per_file = data_type, size_a_image, frame_per_file
        return data_type, size_a_image, frame_per_file

    ##  get offset array
    def __getoffset(self):
        self.size_a_image = self.header[1] * self.header[2]
        frame_total = sum(self.frame_per_file)
        frame_file_max = self.frame_per_file[0]
        offset = []
        fileNumber = []
        a = 0
        b = 0
        for i in range(frame_total):
            offset += [a * self.size_a_image]
            fileNumber += [np.floor(i / frame_file_max).astype(int)]
            if np.floor((i + 1) / frame_file_max) == b:
                a += 1
            else:
                a = 0
                b += 1
        return offset, fileNumber
    ###############################################################################


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