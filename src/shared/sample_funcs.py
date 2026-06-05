import numpy as np

def appendix(x):
    return ((6*x - 2)**2) * np.sin((12*x) - 4)

def peaks(matr):
    x = matr[:,0]
    y = matr[:,1]

    return (3 * (1 - x)**2 * np.exp(-x**2 - (y + 1)**2)
            - 10 * (x/5 - x**3 - y**5) * np.exp(-x**2 - y**2)
            - 1/3 * np.exp(-(x + 1)**2 - y**2))