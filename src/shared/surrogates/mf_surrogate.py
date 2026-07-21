from collections.abc import Callable
from abc import ABC, abstractmethod

import numpy as np


class MFSurrogate(ABC):


    def __init__(self, x: np.ndarray, nlow: int, func_e: Callable, func_c: Callable, verbose: bool = False):
        self._verbose = verbose

        self.x = x

        self.n   = self.x.shape[0]
        self.dim = self.x.shape[1]

        self.nlow = nlow
        self.nhigh = self.n - self.nlow

        self.func_e = func_e
        self.func_c = func_c

        self.x_c = x[:self.nlow]
        self.x_e = x[:self.nhigh]
        
        self.y_c = self.func_c(self.x_c)
        self.y_e = self.func_e(self.x_e)
        

    def __call__(self, *coords):
        """
        Helper to make the fitter's model one-to-one
        swappable with the objective target function used to
        build the model. Does not evaluate uncertainty.

        :param coords: tuple of coordinates,
            this should work regardless of dimension of target function.
        """
        x = np.array(coords)
        y = self.evaluate(x)
        return y


    def test_check_model(self):
        """
        A test that checks the model at data points.
        The value should be close to zero.
        :return:
        """
        epss = []
        us = []
        for i in range(self._n):
            ypred, upred = self.evaluate_uncertainty(self.x[i,:])
            us.append(upred)
                
            yref = self.y[i]
            eps = abs(ypred - yref)
            epss.append(eps)

        mx = max(epss)
        mn = min(epss)

        print(f"minimum error: {mn}")
        print(f"maximum error: {mx}")
        print(f"average error: {np.mean(epss)}")
        print(f"uncertainties: {us}")
        print(f"avg uncertainty: {np.mean(us)}")


    @abstractmethod
    def evaluate_uncertainty(self, x):
        """Return (ypred, upred) for a given input x."""
    
    
    @abstractmethod
    def evaluate(self, x):
        """Return ypred for a given input x."""