from basic.binning import binning, scatter_hist
from EM_Algorithm.gen_gauss import gen_gauss
from EM_Algorithm.gen_poisson import gen_poisson
import numpy as np
import matplotlib.pyplot as plt

x = gen_gauss([8],[2],[1000])
y = gen_poisson([1],[1000])

pdx, centerx, fig_x, ax_x = binning(x,10,show=False)
pdy, centery, fig_y, ax_y = binning(y,10,show=False)

fig = plt.figure(figsize=(8, 8))

# Add a gridspec with two rows and two columns and a ratio of 2 to 7 between
# the size of the marginal axes and the main axes in both directions.
# Also adjust the subplot parameters for a square plot.
gs = fig.add_gridspec(2, 2, width_ratios=(7, 2), height_ratios=(2, 7),
                      left=0.1, right=0.9, bottom=0.1, top=0.9,
                      wspace=0.05, hspace=0.05)

ax = fig.add_subplot(gs[1, 0])
ax.scatter(x, y)
ax_histx = fig.add_subplot(gs[0, 0], sharex=ax)
ax_histy = fig.add_subplot(gs[1, 1], sharey=ax)

ax_histx.hist(x, bins=10, color='grey', edgecolor="white")
ax_histy.hist(y, bins=10, orientation='horizontal', color='grey', edgecolor="white")

