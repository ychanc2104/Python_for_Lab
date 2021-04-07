import time

def timing(func):
    def time_count(*args, **kargs):
        t_start = time.time()
        values = func(*args, **kargs)
        t_end = time.time()
        print (f"{func.__name__} time consuming:  {(t_end - t_start):.3f} seconds")
        return values
    return time_count