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
from TPM.TrackingGlimpse import *
from glob import glob
import os

### parameters for tracking
read_mode = 1 ## mode = 0 is only calculate 'frame_setread_num' frame, other numbers(default) present calculate whole glimpsefile
frame_setread_num = 21 ## only useful when mode = 0, can't exceed frame number of a file
frame_start = 0 ## starting frame for tracking
IC = True

if __name__ == "__main__":
    path_folder = select_folder()
    path_folders = glob(os.path.join(path_folder, '*'))
    for path_folder in path_folders:
        Glimpse_data, Save_df = Analyzing(path_folder, read_mode, frame_setread_num, frame_start, criteria_dist,
                                          aoi_size, frame_read_forcenter, N_loc, contrast, low, high,
                                          blacklevel, whitelevel, put_text, IC)
