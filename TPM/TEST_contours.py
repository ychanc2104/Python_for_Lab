
from TPM.BinaryImage import twoD_Gaussian
from basic.noise import normal_2d
import numpy as np
import cv2
import matplotlib.pyplot as plt

def get_xy(contour):
    s = np.array(contour).shape
    m00 = s[0]
    [[m10, m01]] = np.sum(contour, axis=0)
    cX, cY = m10/m00, m01/m00
    return cX, cY

if __name__ == "__main__":
    low = 30
    high = low*3
    aoi_size = 20
    x = np.array([[i for i in range(20)] for j in range(20)])
    y = np.array([[j for i in range(20)] for j in range(20)])
    params = [42, 2.3, 2.6, 10.5, 11.6, 10, 6.5]
    # image_tofit = twoD_Gaussian([x,y], *params).reshape(20,20) + normal_2d(aoi_size, s=100)
    image_tofit = twoD_Gaussian([x,y], *params).reshape(20,20)

    image = np.uint8(image_tofit)
    edges = cv2.Canny(image, low, high)  # cv2.Canny(image, a, b), reject value < a and detect value > b
    contours, hierarchy = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

    fig, ax = plt.subplots()
    ax.imshow(image)
    fig, ax = plt.subplots()
    ax.imshow(edges)

    perimeters, areas, cX, cY = [], [], [], []
    for c in contours:
        perimeters += [cv2.arcLength(c, True)]
        areas += [cv2.contourArea(c)]
        x, y = get_xy(c)
        M = cv2.moments(c)
        cX += [x]
        cY += [y]
        # if (M['m00'] != 0):
        #     cX += [(M['m10'] / M['m00'])]
        #     cY += [(M['m01'] / M['m00'])]

    print(f'center of position is {cX[0]:.2f}, {cY[0]:.2f}')

