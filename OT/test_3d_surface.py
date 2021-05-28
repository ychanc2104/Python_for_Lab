import math

import numpy as np
import matplotlib.pyplot as plt


def z_function(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

def exp_gauss_2d(x, t, f, tau, m, sigma):
    g = f*np.exp(-t/tau)*sigma/np.sqrt(2*math.pi)*np.exp(-(x-m)**2/(2*sigma**2))
    return g


# x = np.linspace(-6, 6, 100)
# y = np.linspace(-6, 6, 100)
# X, Y = np.meshgrid(x, y)
# Z = z_function(X, Y)

x = np.linspace(0, 20, 100)
t = np.linspace(0, 10, 100)
X, Y = np.meshgrid(x, t)
Z = exp_gauss_2d(X, Y, f=1, tau=1, m=8, sigma=2.5)

fig = plt.figure()
ax = plt.axes(projection="3d")
ax.plot_wireframe(X, Y, Z, color='green')
ax.set_xlabel('step size')
ax.set_ylabel('dwell time')
ax.set_zlabel('pdf')

ax = plt.axes(projection='3d')
ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
                cmap='winter', edgecolor='none')