


import numpy as np

def rbf_linear(r):
    return r

def rbf_cubic(r):
    return r**3

def rbf_thin_plate_spline(r):
    if r < 1e-200:
        return 0
    else:
        return r**2*np.log(r)

def rbf_gaussian(r, sigma = 1):
    return np.exp(-(r**2)/(2*sigma**2))

def rbf_multiquadratic(r, sigma = 1):
    return np.sqrt(r**2 + sigma**2)

def rbf_inverse_multiquadratic(r, sigma = 1):
    return 1.0/rbf_multiquadratic(r, sigma)


class RadialBasisFitter:
    """

    """

    def __init__(self):
        pass

