# -*- coding: utf-8 -*-
"""
1. Localization:
(a) get average image of N_loc pictures
(b) get contours using Canny edge detection algorithm
(c) get edges of contours, and use image moment of edges to get center of positions, (x,y)
(d) get avg. intensity of each aoi, and remove aoi which avg. intensity < blacklevel
(e) sort (x,y) of each aoi according to distance between y-axis(x=0)
(f) select one aoi of each cluster. cluster: all aoi which distance < criteria_dist
(g) fit each aoi with 2D Gaussian to get accurate (x,y)
(h) draw aoi circle and show(save or not) figure to 'output.png'

"""
### import used modules first
from TPM.BinaryImage import BinaryImage
import time
import tkinter as tk
from tkinter import filedialog

put_text = True
criteria_dist = 10  # beabs are closer than 'criteria_dist' will remove
aoi_size = 10
frame_read_forcenter = 55  # no need to change, frame to autocenter beads
N_loc = 40  # number of frame to stack and localization
contrast = 7
low = 40
high = 120
blacklevel = 30
whitelevel = 70
def select_folder():
    root = tk.Tk()
    root.withdraw()
    path_folder = filedialog.askdirectory()
    return path_folder

def localization(path_folder, criteria_dist, aoi_size, frame_read_forcenter,
                 N_loc, contrast, low, high, blacklevel, whitelevel):
    Glimpse_data = BinaryImage(path_folder, criteria_dist=criteria_dist, aoi_size=aoi_size,
                               frame_read_forcenter=frame_read_forcenter, N_loc=N_loc,
                               contrast=contrast, low=low, high=high, blacklevel=blacklevel, whitelevel=whitelevel)
    image, cX, cY = Glimpse_data.Localize(put_text=put_text)  # localize beads
    return image, cX, cY

if __name__ == "__main__":
    path_folder = select_folder()
    t1 = time.time()
    image, cX, cY = localization(path_folder, criteria_dist, aoi_size, frame_read_forcenter,
                                 N_loc, contrast, low, high, blacklevel, whitelevel)
    time_spent = time.time() - t1
    print('spent ' + str(time_spent) + ' s')





