import numpy as np

##  moving average filter
def MA(data, window, mode='silding'):
    data = np.array(data)
    n = len(data)
    data_filter = []
    m = int(window)
    if mode == 'silding':
        for i in range(n):
            if i < m:
                data_filter += [np.mean(data[:(i+1)])]
            else:
                data_filter += [np.mean(data[i-window+1:i+1])]
    return np.array(data_filter)