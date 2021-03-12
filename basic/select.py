
import tkinter as tk
from tkinter import filedialog


def select_folder():
    root = tk.Tk()
    root.withdraw()
    path_folder = filedialog.askdirectory()
    return path_folder

def select_file():
    root = tk.Tk()
    root.withdraw()
    path_file = filedialog.askopenfile()
    return path_file