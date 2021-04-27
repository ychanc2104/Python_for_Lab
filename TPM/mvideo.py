# -*- coding: utf-8 -*-
"""
Flowchart
1. Localization:
(a) get average image of N_loc pictures
(b) get contours using Canny edge detection algorithm
(c) get edges of contours, and use image moment of edges to get center of positions, (x,y)
(d) get avg. intensity of each aoi, and remove aoi which avg. intensity < blacklevel
(e) sort (x,y) of each aoi according to distance between y-axis(x=0)
(f) select one aoi of each cluster. cluster: all aoi which distance < criteria_dist
(g) fit each aoi with 2D Gaussian to get accurate (x,y)
(h) draw aoi circle and show(save or not) figure to 'output.png'

2. Tracking all aoi with 2D Gaussian

3. Save fitting cideo (optional)

4. Save four files for fitting results

"""

### import used modules first
from TPM.localization import *
from TPM.localization import select_folder
import matplotlib.pyplot as plt

selected_aoi = 15
N = 100

read_mode = 0  # mode = 0 is only calculate 'frame_setread_num' frame, other numbers(default) present calculate whole glimpsefile
frame_start = 9000 ## starting frame for tracking
frame_setread_num = N  # only useful when mode = 0, can't exceed frame number of a file

if __name__ == "__main__":
    path_folder = select_folder()

    ### Localization
    Glimpse_data, bead_radius, random_string = localization(path_folder, read_mode, frame_setread_num, frame_start, criteria_dist,
                                             aoi_size, frame_read_forcenter, N_loc, contrast, low, high,
                                             blacklevel, whitelevel, put_text)

    tracking_results = Glimpse_data.Track_All_Frames(selected_aoi, IC=True)
    Glimpse_data.Get_fitting_video_offline(selected_aoi=0, frame_i=frame_start, N=N)


    cX = Glimpse_data.cX[selected_aoi]
    cY = Glimpse_data.cY[selected_aoi]
    image = Glimpse_data.image
    row = int(cY)  # cY, height
    col = int(cX)  # cX, width
    size_half = int(aoi_size / 2)
    image_cut = image[row - size_half:(row + size_half), col - size_half:(col + size_half)]
    image_aoi = Glimpse_data.image_aoi




