
import tkinter as tk
from tkinter import filedialog
import os
from glob import glob
import scipy.io as sio


def select_folder():
    root = tk.Tk()
    root.withdraw()
    path_folder = filedialog.askdirectory()
    return path_folder

def select_file():
    root = tk.Tk()
    root.withdraw()
    path_file = filedialog.askopenfile()
    return path_file.name

def get_files(regex_filename, dialog=True, path_folder=''):
    if dialog==True:
        path_folder = select_folder()
    else:
        path_folder = path_folder
    path_data = glob(os.path.join(path_folder, regex_filename))
    return path_data

def get_mat(path_file=None):
    if path_file == None:
        path_file = select_file()
    data = sio.loadmat(path_file)
    return data

