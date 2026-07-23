import numpy as np


def appendix_one(x):
    return ((6*x - 2)**2) * np.sin((12*x) - 4)


def branin(matr):
    x = matr[:,0]
    y = matr[:,1]

    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return a*(y - b*x**2 + c*x - r)**2 + s*(1 - t)*np.cos(x) + s


def braninAD(matr, A=0.5, C=-1):
    
    f = A*branin(matr)
    f += C

    return f


def peaks(matr):
    x = matr[:,0]
    y = matr[:,1]

    return (3 * (1 - x)**2 * np.exp(-x**2 - (y + 1)**2)
            - 10 * (x/5 - x**3 - y**5) * np.exp(-x**2 - y**2)
            - 1/3 * np.exp(-(x + 1)**2 - y**2))


def peaksAD(matr, A=0.5, B=10, C=-5, D=0):
    f = A*peaks(matr)
    f += C

    return f


def forrester(x, sd=0):
    """
    Forrester function:
    f(x) = (6x - 2)^2 * sin(12x - 4)

    :param x: input vector to be evaluated
    :param sd: standard deviation of observation noise
    :return: outputs of the function
    """
    x = np.atleast_2d(x).reshape(-1, 1)
    fval = ((6 * x[:, 0] - 2) ** 2) * np.sin(12 * x[:, 0] - 4)
    noise = np.random.randn(x.shape[0]) * sd
    return (fval + noise).reshape(-1, 1)


def forrester_low(x, sd=0):
    """
    Low fidelity forrester function approximation:
    f_low(x) = 0.5 * f_high(x) + 10*(x - 0.5) + 5

    :param x: input vector to be evaluated
    :param sd: standard deviation of observation noise at low fidelity
    :return: outputs of the function
    """
    x = np.atleast_2d(x).reshape(-1, 1)
    high_fidelity = forrester(x, 0)
    f = 0.5 * high_fidelity + 10 * (x[:, [0]] - 0.5) + 5
    noise = np.random.randn(x.shape[0], 1) * sd
    return f + noise


def onevar(x):
    """
    One-variable test function.

    :param x: between 0..1
    :return:
    """
    X = 6*x-2
    return X**2*np.sin(2*X)

onevar.dim = 1


def onevarSin(x, A=0.5, B=10, C=-5, D=0):
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
    f = A*np.sin(onevar(x))
    f += B*((x+D) - 0.5)
    f += C

    return f

onevarSin.dim = 1