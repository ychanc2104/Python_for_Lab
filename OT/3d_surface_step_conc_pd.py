from basic.math_fn import gauss
from matplotlib import rcParams
rcParams["font.family"] = "sans-serif"
rcParams["font.sans-serif"] = ["Arial"]
rcParams.update({'font.size': 12})
import math
import numpy as np
import matplotlib.pyplot as plt


def z_function(x, y):
    return np.sin(np.sqrt(x ** 2 + y ** 2))

def exp_gauss_2d(x, t, f, tau, m, sigma):
    g = f*np.exp(-t/tau)*sigma/np.sqrt(2*math.pi)*np.exp(-(x-m)**2/(2*sigma**2))
    return g

def conc_2gauss_2d(x, f, m, sigma):
    g = f[0]*gauss(x, m[0], sigma[0]) + f[1]*gauss(x, m[1], sigma[1])
    # g = f*sigma/np.sqrt(2*math.pi)*np.exp(-(x-m)**2/(2*sigma**2))

    return g



# f_all = [1.00,0.45,0.55,0.95,0.050,0.68,0.32,0.65,0.35,0.99,0.010]
# m_all = [8.10,4.55,8.71,5.82,12.52,4.61,7.85,4.30,7.52,4.92,14.13]
# s_all = [3.08,1.02,2.35,1.71,0.980,1.03,2.32,1.15,2.61,1.69,1.110]
# c_all = [0.00,0.50,0.50,1.00,1.000,1.50,1.50,2.00,2.00,3.00,3.000]
# colors= [ 'b', 'g', 'b', 'g', 'b' , 'g', 'b', 'g', 'b', 'g', 'b' ]

f_all = [1.00,0.45,0.55,0.68,0.32,0.65,0.35,0.99,0.010]
m_all = [8.10,4.55,8.71,4.61,7.85,4.30,7.52,4.92,14.13]
s_all = [3.08,1.02,2.35,1.03,2.32,1.15,2.61,1.69,1.110]
c_all = [0.00,0.50,0.50,1.50,1.50,2.00,2.00,3.00,3.000]
colors= [ 'b', 'g', 'b', 'g', 'b', 'g', 'b', 'g', 'b' ]

mean =  [8.1,6.84,6.14,5.66,5.43,5.04]
c_mean = [0, 0.50,1.00,1.50,2.00,3.00]
fig = plt.figure()
ax = plt.axes(projection="3d")

ax.set_xlabel('Step-size (count)')
ax.set_ylabel('[S5S1] (uM)')
ax.set_zlabel('Probability density (a.u.)')
# ax.set_ylim(-1,4)

x = np.linspace(0, 15, 50)
for i in range(len(f_all)):
    conc = np.ones(len(x))*c_all[i]
    X, Y = np.meshgrid(x, conc)
    # Z = exp_gauss_2d(X, Y, f=1, tau=1, m=8, sigma=2.5)
    Z = f_all[i]*gauss(X, xm=m_all[i], s=s_all[i])
    # ax.plot_surface(X, Y, Z, color=colors[i])
    ax.plot_wireframe(X, Y, Z, color=colors[i])

    ax.plot_surface(X, Y, Z)

    # fig = plt.figure()
    # ax = plt.axes(projection="3d")
    # ax.plot_wireframe(X, Y, Z, color='green')
    # ax.set_xlabel('step size')
    # ax.set_ylabel('conc. (uM)')
    # ax.set_zlabel('pdf')

    # ax = plt.axes(projection='3d')
    # ax.plot_surface(X, Y, Z, rstride=1, cstride=1,
    #                 cmap='winter', edgecolor='none')

for j in range(len(mean)):

    ax.scatter(mean[j], c_mean[j], 0, color='r')