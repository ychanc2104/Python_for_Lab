import io
import numpy as np
import cv2
import matplotlib.pyplot as plt

## define a function which returns an image as numpy array from figure
def get_img_from_fig(fig, dpi=200):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=dpi)
    buf.seek(0)
    img_arr = np.frombuffer(buf.getvalue(), dtype=np.uint8)
    buf.close()
    img = cv2.imdecode(img_arr, 1)
    # img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)  # cv2.COLOR_BGR2RGB
    return img

def save_img(fig, save_path):
    cv2.imwrite(save_path, get_img_from_fig(fig))