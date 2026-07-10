from collections.abc import Callable

import numpy as np


class Surrogate:


    def __init__(self, X_sample: np.ndarray, func: Callable, code: int = 3,  verbose: bool = False):
        self._verbose = verbose
        
        self.func  = func # function to approximate
        self.code  = code # code to pick basis function
        
        self.X_sample = X_sample

        return