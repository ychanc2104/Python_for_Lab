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

from TPM.BinaryImage import BinaryImage
from TPM.DataToSave import DataToSave
from TPM.localization import select_folder
import time


def Analyzing(path_folder, read_mode, frame_setread_num, criteria_dist, aoi_size,
             frame_start, frame_read_forcenter, N_loc, contrast, blacklevel):
    ### Localization
    Glimpse_data = BinaryImage(path_folder, read_mode=read_mode, frame_setread_num=frame_setread_num, criteria_dist=criteria_dist, aoi_size=aoi_size,
                               frame_read_forcenter=frame_read_forcenter, N_loc=N_loc, frame_start=frame_start,
                               contrast=contrast, blacklevel=blacklevel)
    image, cX, cY = Glimpse_data.Localize(put_text=True) # localize beads
    localization_results = Glimpse_data.radius_save
    localization_results = localization_results.reshape((len(localization_results),1))
    ### Tracking
    tracking_results = Glimpse_data.Track_All_Frames()
    ### Saving results
    Save_df = DataToSave(tracking_results, localization_results, path_folder, frame_start=frame_start, med_fps=Glimpse_data.med_fps, window=20, factor_p2n=10000/180)
    Save_df.Save_four_files()
    return Glimpse_data, Save_df



read_mode = 0 # mode = 0 is only calculate 'frame_setread_num' frame, other numbers(default) present calculate whole glimpsefile
frame_setread_num = 50 # only useful when mode = 0, can't exceed frame number of a file
criteria_dist = 20 # beabs are closer than 'criteria_dist' will remove
aoi_size = 20
frame_start = 13 # no need to change, frame to autocenter beads
frame_read_forcenter = 0
N_loc = 40 # number of frame to stack and localization
contrast = 1
blacklevel = 5


if __name__ == "__main__":
    path_folder = select_folder()
    t1 = time.time()
    Glimpse_data, Save_df = Analyzing(path_folder, read_mode, frame_setread_num, criteria_dist, aoi_size,
              frame_start, frame_read_forcenter, N_loc, contrast, blacklevel)
    time_spent = time.time() - t1
    print('spent ' + str(time_spent) + ' s')


