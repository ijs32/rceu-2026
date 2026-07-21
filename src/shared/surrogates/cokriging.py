import math
from collections.abc import Callable

import numpy as np
from scipy.optimize import minimize
from scipy.linalg import solve_triangular

from shared.surrogates.surrogate import Surrogate


class CoKriging(Surrogate):


    def __init__(self, x: np.ndarray, nlow, func: Callable, verbose: bool = False):
        super().__init__(x, func, verbose=verbose)

        self.theta     = self.__find_theta()
        self.thetad, self.rho = self.__find_theta_rho()
        self.sqrtSigma = self.__build_sigma()

        self.ntot = self.x.shape[0]
        self.nlow = nlow
        self.nhigh = self.x.shape[0] - self.nlow

        # global mean and variance
        # Note: global variance can also be called 'sill' or 'process variance'.
        self.globmean = np.zeros([1])
        self.globvar = np.zeros([1])
        self.theta = np.zeros([self._dim])
        # Psi, correlation matrix, and sqrt Psi
        self.Psi = np.zeros([self.nlow, self.nlow])
        self.sqrtPsi = np.zeros([self.nlow, self.nlow])

        # Scipy optimizer result
        self.result  = None
        self.resultd = None

        self.d = np.zeros([self.nhigh])
        self.globmeand = np.zeros([1])
        self.globvard = np.zeros([1])
        self.thetad = np.zeros([self._dim])
        self.Psid = np.zeros([self.nhigh, self.nhigh])
        self.sqrtPsid = None
        # scaling parameter: High = rho*Low + Delta
        self.rho = 1e-3 # needs to be initialized, setting it to min

        # full covariance matrix
        self.Sigma = np.zeros([self.nlow + self.nhigh, self.nlow + self.nhigh])
        self.sqrtSigma = np.zeros([self.nlow + self.nhigh, self.nlow + self.nhigh])
        self.LnDetPsi = None


    def __set_psi_matrices(self, theta, with_retries: bool):

        retry = with_retries
        adapt = 1
        count = 0
        while retry:
            retry = False

            nugget = 1e-11 * adapt
            PSI = np.eye(self.nlow)*(1.0+nugget)

            for i in range(0,self.nlow):
                for j in range(i+1, self.nlow):
                    prod = (self.x[i,:] - self.x[j,:])**2
                    prod = np.dot(theta, prod)
                    # print(f"prod {prod}")
                    cij = np.exp(-prod)
                    PSI[i,j] = cij
                    PSI[j,i] = cij

            sqrtPSI = np.linalg.cholesky(PSI)
            if with_retries and count < 5:
                LnDetPsi = 2*np.sum(np.log(np.abs(np.diagonal(sqrtPSI))))
                if LnDetPsi < -50:
                    adapt *= 100
                    retry = True
                    count += 1
                else:
                    print("ILL CONDITIONED")
                    return 1e10
            
            self.PSI = PSI
            self.sqrtPSI = sqrtPSI

        
    def __find_theta(self):
        # initialize
        thetamin = 1e-3
        thetamax = 1e2

        best_result = None
        min_log_likelihood = math.inf
        for _ in range(5):

            theta0 = 10**np.random.uniform(-3, 2, size=self._dim)
            # find theta using a numerical method
            result = minimize(
                self.__negative_concentrated_log_likelihood,
                theta0,
                bounds=self._dim*[(thetamin,thetamax)],
                method='L-BFGS-B',
            )

            if result.fun < min_log_likelihood:
                min_log_likelihood = result.fun
                best_result = result

        assert best_result is not None, "optimization failed to produce a result"
        if self._verbose:
            print(f"theta {best_result.x}")
            print("FULL best_result", best_result)

        self.__set_psi_matrices(best_result.x, False)
        # for diagnostics
        self.result = best_result

        return best_result.x # theta


    def __negative_concentrated_log_likelihood(self, theta):

        try:
            self.__set_psi_matrices(theta, True)
        except np.linalg.LinAlgError:
            print("LINALGERROR")
            return 1e10
        
        if self._verbose:
            print(f"√PSI {self.sqrtPSI}")

        # find global mean (needed for global variance)
        tmp = solve_triangular(self.sqrtPSI, self.y, lower=True)
        tmp2 = solve_triangular(self.sqrtPSI, np.ones([self.nlow]), lower=True)

        self.globmean = np.dot(tmp, tmp2)
        self.globmean /= np.dot(tmp2, tmp2)
        if self._verbose:
            print(f"globmean {self.globmean} globvar {self.globvar}")

        # find global variance (needed for concentrated log likelihood)
        tmp = solve_triangular(self.sqrtPSI, self.y - self.globmean, lower=True)
        self.globvar = np.dot(tmp, tmp)/self.nlow

        # find concentrated log likelihood
        LnDetPSI = 2*np.sum(np.log(np.abs(np.diagonal(self.sqrtPSI))))
        return self.nlow/2 * np.log(self.globvar) + 0.5 * LnDetPSI
    

    def __find_theta_rho(self):
        # initialize
        theta_rho_min = 1e-3
        theta_rho_max = 1e2

        best_result_d = None
        min_log_likelihood = math.inf
        for _ in range(10):
            theta_rho_0 = 10**np.random.uniform(-3, 2, size=(self._dim+1))

            # find theta using a numerical method
            resultd = minimize(
                self.__negative_concentrated_log_likelihood_d,
                theta_rho_0,
                # todo check the bounds for rho
                bounds=(self._dim+1)*[(theta_rho_min,theta_rho_max)],
                method='L-BFGS-B',
            )

            if resultd.fun < min_log_likelihood:
                min_log_likelihood = resultd.fun
                best_result_d = resultd

        assert best_result_d is not None, "optimization failed to produce a result"
        if self.verbose:
            print(f"theta {best_result_d.x}")
            print("FULL RESULT", best_result_d)

        # for diagnostics
        self.resultd = best_result_d

        # lock in final state after theta is found
        self.__set_psi_d_matrices(best_result_d.x, False)

        thetad = best_result_d.x[:self._dim]
        rho = best_result_d.x[-1]

        return thetad, rho


    def __set_psi_d_matrices(self, theta_rho, with_retries: bool):
        rho = theta_rho[self._dim]
        theta = theta_rho[:self._dim]

        for i in range(self.nhigh):
            self.d[i] = self.y[self.nlow+i] - rho*self.y[i]

        retry = with_retries
        adapt = 1
        count = 0
        while retry:
            retry = False

            nugget = 1e-11 * adapt
            Psid = np.eye(self.nhigh)*(1.0+nugget)
            for i in range(0,self.nhigh):
                for j in range(i+1, self.nhigh):

                    prod = (self.x[i,:] - self.x[j,:])**2
                    prod = np.dot(theta, prod)
                    # print(f"prod {prod}")
                    cij = np.exp(-prod)
                    Psid[i,j] = cij
                    Psid[j,i] = cij

            if self.verbose:
                if self.nhigh < 10:
                    print(f"Psid {Psid}")
                else:
                    print(f"Psid")

            sqrtPsid = np.linalg.cholesky(Psid)
            
            if with_retries and count < 5:
                LnDetPsi = 2*np.sum(np.log(np.abs(np.diagonal(sqrtPsid))))
                if LnDetPsi < -50:
                    adapt *= 100
                    retry = True
                    count += 1
                else:
                    print("ILL CONDITIONED")
                    return 1e10
            
        if self.verbose:
            print(f"√Psid {sqrtPsid}")
            
        self.Psid = Psid
        self.sqrtPsid = sqrtPsid


    def __negative_concentrated_log_likelihood_d(self, theta_rho):
        # HIGH-FIDELITY:
        
        # penalty if ill-conditioned:
        # push away the optimizer but don't stop it.
        try:
            self.__set_psi_d_matrices(theta_rho, True)
        except np.linalg.LinAlgError:
            print("LINALGERROR")
            return 1e10

        # find global mean (needed for global variance)
        tmp = solve_triangular(self.sqrtPsid, self.d, lower=True)
        tmp2 = solve_triangular(self.sqrtPsid, np.ones([self.nhigh]), lower=True)

        # print(f"tmp {tmp} tmp2 {tmp2}")
        self.globmeand = np.dot(tmp, tmp2)
        self.globmeand /= np.dot(tmp2, tmp2)

        # find global variance (needed for concentrated log likelihood)
        tmp = solve_triangular(self.sqrtPsid, self.d - self.globmeand, lower=True)
        self.globvard = np.dot(tmp, tmp)/self.nhigh
        
        # find concentrated log likelihood
        # sic: use sqrtPsi, not sqrtPsid
        LnDetPsi = 2*np.sum(np.log(np.abs(np.diagonal(self.sqrtPsi))))
        return self.nhigh/2 * np.log(self.globvard) + 0.5 * LnDetPsi


    def build_sigma(self):
        # Build Sigma
        nugget = (1.0+1e-11)
        Sigma = np.zeros([self.nlow + self.nhigh, self.nlow + self.nhigh])
        for i in range(0,self.nhigh+self.nlow):

            # Build Diagonal
            if i < self.nlow:
                Sigma[i,i] = (self.globvar) * nugget
            else:
                Sigma[i,i] = (self.rho**2 * self.globvar + self.globvard) * nugget

            for j in range(i+1, self.nhigh+self.nlow):
                if i < self.nlow and j < self.nlow:
                    # top left corner
                    prod = (self.x[i,:] - self.x[j,:])**2
                    prod = np.dot(self.theta, prod)
                    cij = self.globvar * np.exp(-prod)
                    Sigma[i,j] = cij
                    Sigma[j,i] = cij
                elif i >= self.nlow and j >= self.nlow:
                    # high-fidelity subblock -- bottom right corner
                    prodc = (self.x[i,:] - self.x[j,:])**2
                    prodc = np.dot(self.theta, prodc)
                    cijc = self.rho**2 * self.globvar * np.exp(-prodc)

                    prodd = (self.x[i,:] - self.x[j,:])**2
                    prodd = np.dot(self.thetad, prodd)
                    cijd = self.globvard * np.exp(-prodd)
                    Sigma[i,j] = cijc + cijd
                    Sigma[j,i] = cijc + cijd
                else:
                    # bottom left and top right
                    prod = (self.x[i,:] - self.x[j,:])**2
                    prod = np.dot(self.theta, prod)
                    cij = self.rho * self.globvar * np.exp(-prod)
                    Sigma[i,j] = cij
                    Sigma[j,i] = cij
        
        try:
            sqrtSigma = np.linalg.cholesky(Sigma)
        except np.linalg.LinAlgError as e:
            print("LINALGERROR: ",e)
            return 1e10
        
        if self.verbose:
            print(f"√Sigma {sqrtSigma}")

        return sqrtSigma


    def evaluate(self, x):
        """
        Evaluate (also called inference, or prediction)
        the statistical Kriging model on input x.
        No dimension checks. Must first call fit() to initialize
        model parameters.
        :param x:
        :return: y
        """

        # build c
        c = np.zeros((self.ntot))
        for i in range(self.ntot):
            if i < self.nlow:
                # low-fidelity -- cheap points
                prod = (self.x[i,:] - x)**2
                prod = np.dot(self.theta, prod)
                ci = self.rho * self.globvar * np.exp(-prod)
                c[i]= ci
            else:
                # high-fidelity -- expensive points
                prodc = (self.x[i,:] - x)**2
                prodc = np.dot(self.theta, prodc)
                cic = self.rho**2 * self.globvar * np.exp(-prodc)

                prodd = (self.x[i,:] - x)**2
                prodd = np.dot(self.thetad, prodd)
                cid = self.globvard * np.exp(-prodd)
                c[i] = cic + cid

        y = self.globmean
        tmp = solve_triangular(self.sqrtSigma, c, lower=True)
        tmp2 = solve_triangular(self.sqrtSigma, self.y - self.globmean, lower=True)
        y += np.dot(tmp, tmp2)
        return y


    def evaluate_uncertainty(self, x):
        """
        Evaluate and also find the uncertainty at the same point.
        :param x:
        :return: y, u
        """

        # build c
        c = np.zeros((self.ntot))
        for i in range(self.ntot):
            if i < self.nlow:
                # low-fidelity -- cheap points
                prod = (self.x[i,:] - x)**2
                prod = np.dot(self.theta, prod)
                ci = self.rho * self.globvar * np.exp(-prod)
                c[i]= ci
            else:
                # high-fidelity -- expensive points
                prodc = (self.x[i,:] - x)**2
                prodc = np.dot(self.theta, prodc)
                cic = self.rho**2 * self.globvar * np.exp(-prodc)

                prodd = (self.x[i,:] - x)**2
                prodd = np.dot(self.thetad, prodd)
                cid = self.globvard * np.exp(-prodd)
                c[i] = cic + cid


        y = self.globmean
        tmp = solve_triangular(self.sqrtSigma, c, lower=True)
        tmp2 = solve_triangular(self.sqrtSigma, self.y - self.globmean, lower=True)
        y += np.dot(tmp, tmp2)
        
        u = np.sqrt(self.rho**2*self.globvar + self.globvard - np.dot(tmp,tmp))
        return y, u
    

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
    