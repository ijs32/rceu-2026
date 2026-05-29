


import numpy as np
from .functions_data import aerofoil_drag_data

aerofoil_drag_lookup = np.array(aerofoil_drag_data)


class Noisegen:
    """
    A helper class to handle randomness,
    mostly wrapped up in Numpy and using
    Numpy defaults. Set the seed to a fixed integer if reproducibility
    is needed.

    :param seed: positive integer
    """
    def __init__(self, seed = None):
        self.rng = np.random.default_rng(seed)

    def uniform(self, N, a, b, sort=False):
        samples = self.rng.uniform(a, b, N)
        if sort:
            samples.sort()
        return samples

    def normal(self, N, mean = 0.0, stddev = 1.0):
        samples = self.rng.normal(mean, stddev, N)
        return samples



def function_noise_1d(function, xmin, xmax, npoints, stddev):
    """
    Generate a random sample in the given range,
    of given size, with normally-distributed noise
    having the given standard deviation.

    :param function:
    :param xmin:
    :param xmax:
    :param npoints:
    :param stddev:
    :return:
    """
    ngen = Noisegen()
    x = ngen.uniform(npoints, xmin, xmax, sort=True)
    y = function(x)
    noise = ngen.normal(npoints, stddev = stddev)
    y = y + noise
    return x, y



def onevar(x):
    """
    One-variable test function.

    :param x: between 0..1
    :return:
    """
    X = 6*x-2
    return X**2*np.sin(2*X)


def onevarAD(x, A=0.5, B=10, C=-5, D=0):
    """
    Cheap one-variable test function with
    parameters A,B,C,D.

    :param x: between 0..1
    :param A:
    :param B:
    :param C:
    :param D:
    :return: f(x)
    """
    f = A*(((x+D)*6+2)**2)
    f *= np.sin(((x+D)*6-2)*2)
    # ??? why not scale x by 6?
    f += B*((x+D) - 0.5)
    f += C
    return f



def rastrigin(x):
    """
    This function takes a input vector x and
    returns a scalar output of the Rastrigin function,
    a difficult search problem used to test genetic algorithm (GA).
    Global optimum is 0 at x = [0 0 .... 0]
    :param x: any length array, components between 0..1
    :return: scalar Rastrigin function value
    """
    f = 10*len(x) + np.sum(x**2 - 10*np.cos(2*np.pi*x))
    return f


def dome(x):
    """
    Dome test function

    :param x:
    :return:
    """
    out = np.sum(1 - (2*x-1)**2)
    # ???
    out /= len(x)
    return out


def branin(x, y):
    """
    Normalized x, y to Branin function.
    Unnormalized branin: x:-5..10, y:0..15
    :param x: between 0, 1
    :param y: between 0, 1
    :return: branin(x,y)
    """
    X = 15*x-5
    Y = 15*y
    a = 1
    b = 1.5/(4*np.pi**2)
    c = 5/np.pi
    d = 6
    e = 10
    ff = 1/(8*np.pi)
    f = a*(Y - b*X**2 + c*X - d )**2
    f += e*(1 - ff)*np.cos(X)
    f += e
    f += 5*X
    return f


def product_constraint(x):
    """

    :param x:
    :return:
    """
    return np.prod(x)


def aerofoil_drag(Is):
    """
    Arguments;
        Is (1d array): an array of values all 0..1
    """
    # This operation uses the input as
    # indices to a lookup table called `aerofoil_drag_lookup`.
    idxs = np.round(Is*100).astype(int)
    return aerofoil_drag_lookup[idxs]


def liftsurf(
    S_w = 174,
    W_fw = 252,
    A = 7.52,
    Lambda = 0,
    q = 34,
    lambda1 = 0.672,
    tc = 0.12,
    N_z = 3.8,
    W_dg = 2000,
    W_p = 0.064,
):
    """
    Default values are for a
    typical light aircraft value (C172).

    :param S_w: Wing area (ft^2)
    :param W_fw: Weight of fuel in the wing (lb)
    :param A: aspect ratio
    :param Lambda: quarter-chord sweep (deg)
    :param q: dynamic pressure at cruise (lb/ft^2)
    :param lambda1: taper ratio
    :param tc: aerofoil thickness to chord ratio
    :param N_z: ultimate load factor (1.5x limit load factor)
    :param W_dg: flight design gross weight (lb)
    :param W_p: paint weight (lb/ft^2)
    :return W: wing weight [lbs] est. by the eq./actual 245/236
    """
    W = 0.036*S_w**0.758
    W *= W_fw**0.0035
    W *= (A/(np.cos(Lambda)**2))**0.6
    W *= q**0.006
    W *= lambda1**0.04
    W *= (100*tc/np.cos(Lambda))**(-0.3)
    W *= (N_z*W_dg)**0.49
    W += S_w*W_p
    return W


def liftsurf_4d(
        S_w,
        tc,
        N_z,
        W_dg,
):
    """
    Reduced 4-dimensional version of liftsurfw.
    The four most significant variables are active,
    as chosen via the Morris screening algorithm.
    The rest are kept at the baseline value.
    """
    return liftsurf(
        S_w=S_w,
        tc=tc,
        N_z=N_z,
        W_dg=W_dg,
    )


