import math
from collections.abc import Callable

import numpy as np
from scipy.spatial.distance import cdist

from shared.psi import psi

class RBF:

    def __init__(self, X_sample: np.ndarray, func: Callable, code: int):
        self.func  = func # function to approximate
        self.code  = code # code to pick basis function
        
        self.X_sample = X_sample

        return
    
    def get_psi_matrix(self, XA: np.ndarray, XB: np.ndarray, sigma) -> np.ndarray:
        D = cdist(XA, XB)
 
        return psi(D, self.code, sigma)
    
    def get_weights(self, X, PSI):
        Z_sample = self.func(X)

        return np.linalg.solve(PSI, Z_sample)
    
class Parametric(RBF):

    def __init__(self, X_sample: np.ndarray, func: Callable, code: int, n_sigmas: int = 100):
        super().__init__(X_sample, func, code)

        self.log_sigmas = np.logspace(-3,3,n_sigmas,base=10)

    def optimizer(self, k = 5) -> float:

        best_sigma = 0
        best_loss  = math.inf
        folds = np.array_split(self.X_sample, k)

        for sigma in self.log_sigmas:

            loss  = 0

            for i in range(k):
                X_val = folds[i]
                X_train = np.concatenate([folds[j] for j in range(k) if j != i])

                PSI_sample = self.get_psi_matrix(X_train, X_train, sigma)

                try:
                    W = self.get_weights(X_train, PSI_sample)
                except np.linalg.LinAlgError:
                    loss = np.inf
                    break 

                PSI = self.get_psi_matrix(X_val, X_train, sigma)

                Z_pred = (PSI @ W)
                Z_actual = self.func(X_val)

                loss += np.sum((Z_pred - Z_actual)**2)


            if loss < best_loss:
                best_sigma = sigma
                best_loss  = loss
            
        return best_sigma


"""
def cv_error(log_sigma):
    sigma = 10**log_sigma          # search in log-space
    total = 0
    for each fold (or each left-out point):
        - build PSI on the *training* subset using this sigma
        - solve PSI @ w = y_train        # the "lower level" repair step
        - predict at the *held-out* points
        - accumulate squared error vs. their true y
    return total / n

result = minimize_scalar(cv_error, bounds=(lo, hi), method='bounded')
best_sigma = 10**result.x
"""