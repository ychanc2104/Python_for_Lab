from TPM.BinaryImage import twoD_Gaussian
from basic.noise import normal_2d
import scipy.optimize as opt
import numpy as np
import matplotlib.pyplot as plt



def get_guess(image_tofit):
    aoi_size = 20
    amp_guess = np.max(image_tofit)
    x_guess = np.argmax(image_tofit) % aoi_size
    y_guess = np.argmax(image_tofit) // aoi_size
    background = np.mean(image_tofit)
    initial_guess = [amp_guess, 10, 10, x_guess, y_guess, 0, background]
    return initial_guess


aoi_size = 20
x = np.array([[i for i in range(20)] for j in range(20)])
y = np.array([[j for i in range(20)] for j in range(20)])

params = [42, 2.3, 2.6, 10.5, 11.6, 10, 6.5]
# image_tofit = twoD_Gaussian([x,y], *params).reshape(20,20) + normal_2d(aoi_size, s=5)
image_tofit = twoD_Gaussian([x,y], *params).reshape(20,20)
bounds = ((1, 0.5, 0.5, 0, 0, 0, 0), (255, aoi_size/2, aoi_size/2, aoi_size-1, aoi_size-1, 90, 255))


initial_guess = [80,2,2,5,5,0,0]
initial_guess = get_guess(image_tofit)
popt, pcov = opt.curve_fit(twoD_Gaussian, [x, y], image_tofit.ravel(), initial_guess,
                           bounds=bounds, method='trf')

data_fitted = twoD_Gaussian((x, y), *popt)
fig, ax = plt.subplots(1, 1)
# ax.imshow(image_aoi, cmap=plt.cm.gray, origin='lower',
#           extent=(x.min(), x.max(), y.min(), y.max()))
ax.imshow(image_tofit)

ax.contour(x, y, data_fitted.reshape(aoi_size, aoi_size), 5, colors='r')



