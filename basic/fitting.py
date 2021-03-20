from scipy.optimize import curve_fit

def linear_eq(x, slope, intercept):
    return x*slope + intercept


def L_fit(xdata, ydata):
    popt, pcov = curve_fit(linear_eq, xdata, ydata)
    return popt