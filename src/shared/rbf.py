import math
from collections.abc import Callable

import numpy as np
from scipy.spatial.distance import cdist
from scipy.optimize import minimize
from scipy.linalg import solve_triangular

from shared.psi import psi

class RBF:

    def __init__(self, X_sample: np.ndarray, func: Callable, code: int,  verbose: bool = False):
        self.__verbose = verbose
        
        self.func  = func # function to approximate
        self.code  = code # code to pick basis function
        
        self.X_sample = X_sample

        return
    
    def __get_psi_matrix(self, XA: np.ndarray, XB: np.ndarray, sigma) -> np.ndarray:
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

                PSI_sample = self.__get_psi_matrix(X_train, X_train, sigma)

                try:
                    W = self.get_weights(X_train, PSI_sample)
                except np.linalg.LinAlgError:
                    loss = np.inf
                    break 

                PSI = self.__get_psi_matrix(X_val, X_train, sigma)

                Z_pred = (PSI @ W)
                Z_actual = self.func(X_val)

                loss += np.sum((Z_pred - Z_actual)**2)


            if loss < best_loss:
                best_sigma = sigma
                best_loss  = loss
            
        return best_sigma
    
class Kriging(RBF):

    def __init__(self, X_sample: np.ndarray, func: Callable):
        code = 7 # kriging code for psi func
        super().__init__(X_sample, func, code)

        self.__dim = X_sample.shape[1]
        self.__theta = self.__find_theta()


    def __get_psi_matrix(self, XA: np.ndarray, XB: np.ndarray, theta) -> np.ndarray:
        D = cdist(XA, XB)
 
        return psi(D, self.code, theta=theta)
    

    def __find_theta(self):
        # initialize
        thetamin = 1e-3
        thetamax = 1e2
        theta0 = np.full([self.__dim], 0.1)
        # find theta using a numerical method
        result = minimize(
            self.__negative_concentrated_log_likelihood,
            theta0,
            bounds=self.__dim*[(thetamin,thetamax)],
            method='L-BFGS-B',
        )
        if self.__verbose:
            print(f"theta {result.x}")
            print("FULL RESULT", result)
        # re-run to lock in final state after theta is found
        self.__set_psi_matrices(result.x)
        # for diagnostics
        self.result = result

        return result.x # theta


    def __negative_concentrated_log_likelihood(self, theta):

        try:
            self.__set_psi_matrices(theta)
        except np.linalg.LinAlgError:
            print("LINALGERROR")
            return 1e10
        
        if self.__verbose:
            print(f"√PSI {self.sqrtPSI}")
        # find global mean (needed for global variance)
        tmp = solve_triangular(self.sqrtPSI, self.y, lower=True)
        tmp2 = solve_triangular(self.sqrtPSI, np.ones([self.n]), lower=True)
        # print(f"tmp {tmp} tmp2 {tmp2}")
        self.globmean = np.dot(tmp, tmp2)
        self.globmean /= np.dot(tmp2, tmp2)
        if self.__verbose:
            print(f"globmean {self.globmean} globvar {self.globvar}")
        # find global variance (needed for concentrated log likelihood)
        tmp = solve_triangular(self.sqrtPSI, self.y - self.globmean, lower=True)
        self.globvar = np.dot(tmp, tmp)/self.n
        # find concentrated log likelihood
        LnDetPSI = 2*np.sum(np.log(np.abs(np.diagonal(self.sqrtPSI))))
        return self.n/2 * np.log(self.globvar) + 0.5 * LnDetPSI
    

    def __set_psi_matrices(self, theta):
        self.PSI = self.__get_psi_matrix(
            self.X_sample,
            self.X_sample,
            theta
        )

        self.sqrtPSI = np.linalg.cholesky(self.PSI)


    def predict(self):
        raise NotImplementedError