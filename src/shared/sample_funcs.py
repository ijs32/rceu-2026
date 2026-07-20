import numpy as np


def appendix_one(x):
    return ((6*x - 2)**2) * np.sin((12*x) - 4)


def branin(x,y):

    a = 1.0
    b = 5.1 / (4.0 * np.pi**2)
    c = 5.0 / np.pi
    r = 6.0
    s = 10.0
    t = 1.0 / (8.0 * np.pi)
    return a*(y - b*x**2 + c*x - r)**2 + s*(1 - t)*np.cos(x) + s


def braninAD(x,y, A=0.5, C=-1):
    f = A*branin(x,y)
    f += C

    return f


def peaks(x,y):
    
    return (3 * (1 - x)**2 * np.exp(-x**2 - (y + 1)**2)
            - 10 * (x/5 - x**3 - y**5) * np.exp(-x**2 - y**2)
            - 1/3 * np.exp(-(x + 1)**2 - y**2))


def peaksAD(x,y, A=1, B=10, C=-1, D=0):
    f = A*peaks(x,y)
    f += C

    return f