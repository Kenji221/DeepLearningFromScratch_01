import numpy as np

def mean_squared_error(y,t):
    return np.sum((y-t)**2)*0.5

# 交差エントロピー誤差
def cross_entropy_error(y, t):
    delta = 1e-7
    return -np.sum(t * np.log(y + delta))

