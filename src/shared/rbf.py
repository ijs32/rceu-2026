from collections.abc import Callable

import numpy as np
from scipy.spatial.distance import cdist

from shared.psi import psi

class RBF:

    def __init__(self, X_sample: np.ndarray, X_pred: np.ndarray, func: Callable, code: int):
        self.func            = func # function to approximate
        self.code            = code # code to pick basis function
        self.sigma: float    = 0
        
        self.X_sample: np.ndarray   = X_sample
        self.PSI_sample: np.ndarray = None
        self.W: np.ndarray          = None

        self.X_pred          = X_pred
        self.PSI: np.ndarray = None
        self.Z_pred          = None

        self.PSI_sample = self.get_psi_matrix(X_sample, X_sample)
        self.W          = self.get_weights()

        self.PSI    = self.get_psi_matrix(X_pred, X_sample)
        self.Z_pred = self.get_prediction()

        return
    
    def get_psi_matrix(self, XA: np.ndarray, XB: np.ndarray) -> np.ndarray:
        D = cdist(XA, XB)
 
        return psi(D, self.code)
    
    def get_weights(self):
        Z_sample = self.func(self.X_sample)

        return np.linalg.solve(self.PSI_sample, Z_sample)
    
    def get_prediction(self):
        return (self.PSI @ self.W).reshape((200, 200))


        