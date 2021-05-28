import numpy as np
import math

def to_1darray(*args):
    output = []
    for arg in args:
        output += [np.array(arg, ndmin=1)]
    return output

def gauss(x, xm, s):
    y = 1/s/np.sqrt(2*math.pi)*np.exp(-(x-xm)**2/2/s**2)
    return y

def ln_gauss(x, xm ,s):
    y = - np.log(s) - 1/2*np.log(2*math.pi) - (x-xm)**2/2/s**2
    return y

def exp_dist(t, tau):
    y = 1/tau * np.exp(-t/tau)
    return y

def ln_exp_dist(t, tau):
    lny = -np.log(tau) - t/tau
    return lny

##  args: list of parameters, x: np array for EM
def oneD_gaussian(x, args):
    """Calculate prior probability of each data point
    Parameters
    ----------
    x: (n,)
    args[i]: (k,)

    Returns
    -------
    output: ndarray, shape = (k,n)
    """
    x, f, xm, s = to_1darray(x, args[0], args[1], args[2])
    x = x.ravel()
    y = np.empty((f.size, x.size))
    for i in range(f.size):
        y[i, :] = f[i] * gauss(x, xm[i], s[i])
    return y

def ln_oneD_gaussian(x, args):
    x, f, xm, s = to_1darray(x, args[0], args[1], args[2])
    x = x.ravel()
    lny = np.empty((f.size, x.size))
    for i in range(f.size):
        lny[i, :] = np.log(f[i]) + ln_gauss(x, xm[i], s[i])
    return lny

def exp_survival(t, args):
    t, f, tau = to_1darray(t, args[0], args[1])
    t = t.ravel()
    y = np.empty((f.size, t.size))
    for i in range(f.size):
        y[i, :] = f[i] * np.exp(-t/tau[i])
    return y

##  args: list
def exp_pdf(t, args):
    t, f, tau = to_1darray(t, args[0], args[1])
    t = t.ravel()
    y = np.empty((f.size, t.size))
    for i in range(f.size):
        y[i, :] = f[i] / tau[i] * np.exp(-t / tau[i])
    return y

def ln_exp_pdf(t, args):
    t, f, tau = to_1darray(t, args[0], args[1])
    t = t.ravel()
    lny = np.empty((f.size, t.size))
    for i in range(f.size):
        lny[i, :] = np.log(f[i]) - np.log(tau[i]) - t/tau[i]
    return lny

def gau_exp_pdf(data, args):
    data = data.reshape(-1,2)
    x, t, f, xm, s, tau = to_1darray(data[:, 0], data[:, 1], args[0], args[1], args[2], args[3])
    x = x.ravel()
    t = t.ravel()
    y = np.empty((f.size, x.size))
    for i in range(f.size):
        y[i, :] = f[i] * gauss(x, xm[i], s[i]) * exp_dist(t, tau[i])
    return y

def ln_gau_exp_pdf(data, args):
    data = data.reshape(-1,2)
    x, t, f, xm, s, tau = to_1darray(data[:, 0], data[:, 1], args[0], args[1], args[2], args[3])
    x = x.ravel()
    t = t.ravel()
    lny = np.empty((f.size, x.size))
    for i in range(f.size):
        lny[i, :] = np.log(f[i]) + ln_gauss(x, xm[i], s[i]) + ln_exp_dist(t, tau[i])
    return lny

def exp_gauss_2d(x, t, f, m, sigma, tau):
    g = f*np.exp(-t/tau)*sigma/np.sqrt(2*math.pi)*np.exp(-(x-m)**2/(2*sigma**2))
    return g