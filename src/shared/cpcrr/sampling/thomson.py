

import numpy as np

def get_thomson(n_points, dimension, center):
    fname = 'thomson/thomson'
    if center:
        fname += '.center'
    fname += f'.d{dimension}.n{n_points}.dat'
    X = np.loadtxt(fname,delimiter=',')
    return X
