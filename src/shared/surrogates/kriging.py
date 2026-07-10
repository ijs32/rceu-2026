import math
from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular

from shared.surrogates.surrogate import Surrogate

class Kriging(Surrogate):


    def __init__(self, X_sample: np.ndarray, func: Callable, verbose: bool = False):
        super().__init__(X_sample, func, verbose=verbose)

        self.Y_sample  = self.func(self.X_sample)
        self.__n       = self.X_sample.shape[0]
        self.__dim     = self.X_sample.shape[1]
        self.__theta   = self.__find_theta()
    

    def __set_psi_matrices(self, theta):

        self.PSI = np.eye(self.__n)*(1.0+1e-11)

        for i in range(0,self.__n):
            for j in range(i+1, self.__n):
                prod = (self.X_sample[i,:] - self.X_sample[j,:])**2
                prod = np.dot(theta, prod)
                # print(f"prod {prod}")
                cij = np.exp(-prod)
                self.PSI[i,j] = cij
                self.PSI[j,i] = cij

        self.sqrtPSI = np.linalg.cholesky(self.PSI)

        
    def __find_theta(self):
        # initialize
        thetamin = 1e-3
        thetamax = 1e2

        best_result = None
        min_log_likelihood = math.inf
        for _ in range(5):

            theta0 = 10**np.random.uniform(-3, 2, size=self.__dim)
            # find theta using a numerical method
            result = minimize(
                self.__negative_concentrated_log_likelihood,
                theta0,
                bounds=self.__dim*[(thetamin,thetamax)],
                method='L-BFGS-B',
            )

            if result.fun < min_log_likelihood:
                min_log_likelihood = result.fun
                best_result = result

        assert best_result is not None, "optimization failed to produce a result"
        if self._verbose:
            print(f"theta {best_result.x}")
            print("FULL best_result", best_result)

        self.__set_psi_matrices(best_result.x)
        # for diagnostics
        self.result = best_result

        return best_result.x # theta


    def __negative_concentrated_log_likelihood(self, theta):

        try:
            self.__set_psi_matrices(theta)
        except np.linalg.LinAlgError:
            print("LINALGERROR")
            return 1e10
        
        if self._verbose:
            print(f"√PSI {self.sqrtPSI}")
        # find global mean (needed for global variance)
        tmp = solve_triangular(self.sqrtPSI, self.Y_sample, lower=True)
        tmp2 = solve_triangular(self.sqrtPSI, np.ones([self.__n]), lower=True)
        # print(f"tmp {tmp} tmp2 {tmp2}")
        self.globmean = np.dot(tmp, tmp2)
        self.globmean /= np.dot(tmp2, tmp2)
        if self._verbose:
            print(f"globmean {self.globmean} globvar {self.globvar}")
        # find global variance (needed for concentrated log likelihood)
        tmp = solve_triangular(self.sqrtPSI, self.Y_sample - self.globmean, lower=True)
        self.globvar = np.dot(tmp, tmp)/self.__n
        # find concentrated log likelihood
        LnDetPSI = 2*np.sum(np.log(np.abs(np.diagonal(self.sqrtPSI))))
        return self.__n/2 * np.log(self.globvar) + 0.5 * LnDetPSI
    

    def evaluate(self, x):
        """
        Evaluate (also called inference, or prediction)
        the statistical Kriging model on input x.
        No dimension checks. Must first call fit() to initialize
        model parameters.
        :param x:
        :return: y
        """
        # build psi
        psi = np.zeros([self.__n])
        for i in range(self.__n):
            prod = (self.X_sample[i,:] - x)**2
            prod = np.dot(self.__theta, prod)
            cij = np.exp(-prod)
            psi[i] = cij

        y = self.globmean
        tmp = solve_triangular(self.sqrtPSI, psi, lower=True)
        tmp2 = solve_triangular(self.sqrtPSI, self.Y_sample - y, lower=True)
        y += np.dot(tmp, tmp2)
        return y
    