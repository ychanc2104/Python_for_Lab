import numpy as np

##  moving average filter
def MA(data, window, mode='sliding'):
    data = np.array(data)
    n = len(data)
    data_filter = []
    m = int(window)

    if mode == 'sliding':
        for i in range(n):
            if i < m:
                data_filter += [np.mean(data[:(i+1)])]
            else:
                data_filter += [np.mean(data[i-window+1:i+1])]
    elif mode == 'fixing':
        iteration = n//window
        for i in range(iteration):
            data_filter += [np.mean(data[i*window:(i+1)*window])]

    return np.array(data_filter)