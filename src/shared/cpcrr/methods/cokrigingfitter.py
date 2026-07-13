from collections.abc import Callable
import math

import numpy as np

import matplotlib.pyplot as plt

from scipy.linalg import solve_triangular
from scipy.optimize import minimize



class CoKrigingFitter:
    """
    State manager for co-Kriting fitting experiments.

    :param xmin: vector range (xmin, xmax)
    :param xmax: vector range (xmin, xmax)
    :param datax: array of x-values
    :param datay: 1d array of y-values
    :param nlow: the number of low-fidelity points

    """

    def __init__(self, xmin, xmax, datax, datay, nlow, ymin = None, ymax = None):

        # todo check:
        # nlow ≥ nhigh
        #
        # the nhigh datax values are exactly the same as the nhigh datax values.
        # (Assumption for Kennedy-O'Hagan auto-regressive model)

        if nlow <= (datax.shape[0] / 2):
            raise ValueError("Number of low fidelity points must be greater than the number of high fidelity points.")

        self.xmin = xmin
        self.xmax = xmax
        M, m = datay.max(), datay.min()
        extent = (M-m)
        self.ymin = ymin if ymin is not None else m-0.1*extent
        self.ymax = ymax if ymax is not None else M+0.1*extent
        self.x = datax
        self.y = datay
        if len(self.x.shape) == 1:
            self.x = self.x.reshape(-1,1)
        # dimension of x domain
        self.dim = self.x.shape[1]
        # We Krige the low-fidelity dataset exactly like ordinary Kriging (cf. KrigingFitter)
        self.ntot = self.x.shape[0]
        self.nlow = nlow
        self.n = self.nlow # this confuses me.
        self.nhigh = self.x.shape[0] - self.nlow

        # global mean and variance
        # Note: global variance can also be called 'sill' or 'process variance'.
        self.globmean = np.zeros([1])
        self.globvar = np.zeros([1])
        self.theta = np.zeros([self.dim])
        # Psi, correlation matrix, and sqrt Psi
        self.Psi = np.zeros([self.n, self.n])
        self.sqrtPsi = np.zeros([self.n, self.n])
        self.LnDetPsi = None
        self.fig = plt.figure(figsize=(6, 5), dpi=100)

        # Scipy optimizer result
        self.result  = None
        self.resultd = None


        self.d = np.zeros([self.nhigh])
        self.globmeand = np.zeros([1])
        self.globvard = np.zeros([1])
        self.thetad = np.zeros([self.dim])
        self.Psid = np.zeros([self.nhigh, self.nhigh])
        self.sqrtPsid = None
        # scaling parameter: High = rho*Low + Delta
        self.rho = 1e-3 # needs to be initialized, setting it to min

        # full covariance matrix
        self.Sigma = np.zeros([self.nlow + self.nhigh, self.nlow + self.nhigh])
        self.sqrtSigma = np.zeros([self.nlow + self.nhigh, self.nlow + self.nhigh])



    def fit(self, verbose = False):
        """
        For the given data set (x,y), perform a Kriging fit.

        :param verbose: verbosity
        """
        # LOW-FIDELITY:
        def negative_concentrated_log_likelihood(theta):
            # need to construct Psi in terms of the argument theta,
            # so that the minimizer can iteratively update theta.
            Psi = np.eye(self.nlow)*(1.0+1e-11)
            for i in range(0,self.nlow):
                for j in range(i+1, self.nlow):
                    prod = (self.x[i,:] - self.x[j,:])**2
                    prod = np.dot(theta, prod)
                    # print(f"prod {prod}")
                    cij = np.exp(-prod)
                    Psi[i,j] = cij
                    Psi[j,i] = cij
            if verbose:
                if self.n < 10:
                    print(f"Psi {Psi}")
                else:
                    print(f"Psi")
            # penalty if ill-conditioned:
            # push away the optimizer but don't stop it.
            try:
                sqrtPsi = np.linalg.cholesky(Psi)
            except np.linalg.LinAlgError:
                print("LINALGERROR")
                return 1e10
            if verbose:
                print(f"√Psi {sqrtPsi}")
            self.Psi = Psi
            self.sqrtPsi = sqrtPsi

            # find global mean (needed for global variance)
            tmp = solve_triangular(self.sqrtPsi, self.y[:self.nlow], lower=True)
            tmp2 = solve_triangular(self.sqrtPsi, np.ones([self.n]), lower=True)
            # print(f"tmp {tmp} tmp2 {tmp2}")
            self.globmean = np.dot(tmp, tmp2)
            self.globmean /= np.dot(tmp2, tmp2)
            # find global variance (needed for concentrated log likelihood)
            tmp = solve_triangular(self.sqrtPsi, self.y[:self.nlow] - self.globmean, lower=True)
            self.globvar = np.dot(tmp, tmp)/self.n
            if verbose:
                print(f"globmean {self.globmean} globvar {self.globvar}")
            # find concentrated log likelihood
            LnDetPsi = 2*np.sum(np.log(np.abs(np.diagonal(self.sqrtPsi))))
            self.LnDetPsi = LnDetPsi
            return self.n/2 * np.log(self.globvar) + 0.5 * LnDetPsi

        # initialize
        thetamin = 1e-3
        thetamax = 1e2

        best_result = None
        min_log_likelihood = math.inf
        for _ in range(10):
            theta0 = 10**np.random.uniform(-3, 2, size=self.dim)
            # find theta using a numerical method
            result = minimize(
                negative_concentrated_log_likelihood,
                theta0,
                bounds=self.dim*[(thetamin,thetamax)],
                method='L-BFGS-B',
            )
            
            if result.fun < min_log_likelihood:
                min_log_likelihood = result.fun
                best_result = result

        assert best_result is not None, "optimization failed to produce a result"
        if verbose:
            print(f"theta {best_result.x}")
            print("FULL RESULT", best_result)

        # re-run to lock in final state after theta is found
        negative_concentrated_log_likelihood(best_result.x)
        self.theta = best_result.x
        
        # for diagnostics
        self.result = best_result


        def negative_concentrated_log_likelihood_high(theta_rho):
            rho = theta_rho[self.dim]
            theta = theta_rho[:self.dim]

            # HIGH-FIDELITY:
            for i in range(self.nhigh):
                self.d[i] = self.y[self.nlow+i] - rho*self.y[i]

            Psid = np.eye(self.nhigh)*(1.0+1e-11)
            for i in range(0,self.nhigh):
                for j in range(i+1, self.nhigh):

                    prod = (self.x[i,:] - self.x[j,:])**2
                    prod = np.dot(theta, prod)
                    # print(f"prod {prod}")
                    cij = np.exp(-prod)
                    Psid[i,j] = cij
                    Psid[j,i] = cij
            if verbose:
                if self.n < 10:
                    print(f"Psid {Psid}")
                else:
                    print(f"Psid")
            # penalty if ill-conditioned:
            # push away the optimizer but don't stop it.
            try:
                sqrtPsid = np.linalg.cholesky(Psid)
            except np.linalg.LinAlgError:
                print("LINALGERROR")
                return 1e10
            if verbose:
                print(f"√Psid {sqrtPsid}")

            self.Psid = Psid
            self.sqrtPsid = sqrtPsid

            # find global mean (needed for global variance)
            tmp = solve_triangular(self.sqrtPsid, self.d, lower=True)
            tmp2 = solve_triangular(self.sqrtPsid, np.ones([self.nhigh]), lower=True)
            # print(f"tmp {tmp} tmp2 {tmp2}")
            self.globmeand = np.dot(tmp, tmp2)
            self.globmeand /= np.dot(tmp2, tmp2)
            # find global variance (needed for concentrated log likelihood)
            tmp = solve_triangular(sqrtPsid, self.d - self.globmeand, lower=True)
            self.globvard = np.dot(tmp, tmp)/self.nhigh
            # find concentrated log likelihood
            # sic: use sqrtPsi, not sqrtPsid
            LnDetPsi = 2*np.sum(np.log(np.abs(np.diagonal(self.sqrtPsi))))
            return self.nhigh/2 * np.log(self.globvard) + 0.5 * LnDetPsi

        # initialize
        theta_rho_min = 1e-3
        theta_rho_max = 1e2

        best_result_d = None
        min_log_likelihood = math.inf
        for _ in range(10):
            theta_rho_0 = 10**np.random.uniform(-3, 2, size=(self.dim+1))

            # find theta using a numerical method
            resultd = minimize(
                negative_concentrated_log_likelihood_high,
                theta_rho_0,
                # todo check the bounds for rho
                bounds=(self.dim+1)*[(theta_rho_min,theta_rho_max)],
                method='L-BFGS-B',
            )

            if resultd.fun < min_log_likelihood:
                min_log_likelihood = resultd.fun
                best_result_d = resultd

        assert best_result_d is not None, "optimization failed to produce a result"
        if verbose:
            print(f"theta {best_result_d.x}")
            print("FULL RESULT", best_result_d)

        # re-run to lock in final state after theta is found
        negative_concentrated_log_likelihood_high(best_result_d.x)
        self.thetad = best_result_d.x[:self.dim]
        self.rho = best_result_d.x[-1]

        # for diagnostics
        self.resultd = best_result_d

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
        
        if verbose:
            print(f"√Sigma {sqrtSigma}")

        self.Sigma = Sigma
        self.sqrtSigma = sqrtSigma


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


        # this aint right just fyi. TODO
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


    # Testing

    def test_check_model(self):
        """
        A test that checks the model at data points.
        The value should be close to zero.
        :return:
        """
        epss = []
        us = []
        for i in range(self.n):
            ypred, upred = self.evaluate_uncertainty(self.x[i,:])
            yref = self.y[i]
            eps = abs(ypred - yref)
            epss.append(eps)
            us.append(upred)
        avg = 0.0
        for eps in epss:
            avg += eps
        avg /= self.n
        mx = max(epss)
        mn = min(epss)
        # print(epss)
        print(f"minimum error: {mn}")
        print(f"average error: {avg}")
        print(f"maximum error: {mx}")
        print(f"uncertainties: {us}")


    # Plotting


    def plot_check_inputs(self, objective, res = 100):
        """
        Produce a plot to inspect model
        :param objective: callable objective function
        :param res: resolution (sample values per dimension)
        """
        if self.dim == 1:
            Nside = res
            x = np.linspace(self.xmin[0], self.xmax[0], Nside)
            y = objective(x)
            plt.plot(x, y, color='black', linewidth=0.5)
            plt.scatter(self.x, self.y, color='black', marker='^', s=7.5, alpha=1.0)
            plt.xlim((self.xmin[0], self.xmax[0]))
            plt.title(f"$f(x)$ ($n={len(self.x)}$)")
        elif self.dim == 2:
            Nside = res
            x0 = np.linspace(self.xmin[0], self.xmax[0], Nside)
            y0 = np.linspace(self.xmin[1], self.xmax[1], Nside)
            X, Y = np.meshgrid(x0, y0)
            datax_plot = np.column_stack((X.reshape(-1), Y.reshape(-1)))
            Z = objective(datax_plot[:,0], datax_plot[:,1]).reshape((Nside, Nside))
            contour = plt.contourf(X, Y, Z, levels=24, cmap='viridis', alpha=0.85)
            plt.scatter(self.x[:, 0], self.x[:, 1], color='black', marker='^', s=7.5, alpha=1.0)
            plt.colorbar(contour, label='$f(x,y)$')
            plt.grid(True)
            plt.title(f"$f(x,y)$ ($n={len(self.x)}$)")
        else:
            raise NotImplementedError
        plt.show()

    def plot_check_model(self, objective_e: Callable, objective_c: Callable | None = None, res = 100):
        """
        Produce a plot to inspect model
        :param res: resolution (sample values per dimension)
        """
        if self.dim ==1:
            # for 1d case we also plot uncertainty
            Nside = res
            x = np.linspace(self.xmin[0], self.xmax[0], Nside)
            y = np.ones(Nside)
            u = np.ones(Nside)
            for i in range(Nside):
                y[i], u[i] = self.evaluate_uncertainty(x[i])
                
            plt.fill_between(
                x, y-u, y+u,
                color='#E0E4E8',
                alpha=0.6,
                edgecolor='none',
                label=r'$\pm\sigma$',
            ),
            plt.plot(
                x, y,
                color='blue',
                alpha=0.6,
                linewidth=0.9,
                linestyle='--',
                label=r'$\hat{y}$',
                zorder=3,
            )
            plt.plot(
                x, objective_e(x),
                color='black',
                alpha=0.6,
                linewidth=0.9,
                linestyle='-',
                label='$y_e$',
                zorder=3,
            )
            plt.plot(
                x, objective_c(x),
                color='red',
                alpha=0.6,
                linewidth=0.9,
                linestyle='--',
                label='$y_c$',
                zorder=3,
            )
            scatter_config1 = {
                'color':'black',
                'marker':'^',
                's':10,
                'alpha':1.0,
                'label':'$X_e,Y_e$',
                'clip_on':False,
            }
            scatter_config2 = {
                'color':'red',
                's':20,
                # 'edgecolors':'#ffffff',
                'linewidths':1.1,
                'zorder':4,
                'label':'$X_c,Y_c$',
                'clip_on':False,
            }
            plt.scatter(
                self.x[:self.nlow], self.y[:self.nlow],
                **scatter_config2,
            )
            plt.scatter(
                self.x[self.nlow:self.ntot], self.y[self.nlow:self.ntot],
                **scatter_config1,
            )
            plt.xlim((self.xmin[0], self.xmax[0]))
            plt.xlabel('$x$')
            plt.ylabel('response')
            plt.grid(True, linestyle='--', linewidth=0.5, color='#BDC3C7', alpha=0.7, zorder=0)
            plt.gca().spines['top'].set_visible(False)
            plt.gca().spines['right'].set_visible(False)
            plt.gca().spines['left'].set_color('#BDC3C7')
            plt.gca().spines['bottom'].set_color('#BDC3C7')
            plt.tick_params(axis='both', colors='#7F8C8D', labelsize=10)
            plt.legend(
                # loc='upper right',
                frameon=True,
                facecolor='#ffffff',
                edgecolor='none',      # Frameless legend look
                fontsize=10
            )
            # plt.tight_layout()
            plt.title(rf"co-Kriging $\hat{{y}}$ ($n={len(self.x)}$)")
        elif self.dim == 2:
            Nside = res
            x0 = np.linspace(self.xmin[0], self.xmax[0], Nside)
            y0 = np.linspace(self.xmin[1], self.xmax[1], Nside)
            X, Y = np.meshgrid(x0, y0)
            Z = np.ones([Nside*Nside])
            datax_plot = np.column_stack((X.reshape(-1), Y.reshape(-1)))
            for i in range(datax_plot.shape[0]):
                Z[i] = self.evaluate(datax_plot[i,:])
            Z = Z.reshape((Nside, Nside))
            contour = plt.contourf(X, Y, Z, levels=24, cmap='viridis', alpha=0.85)
            # plot collocation points used to perform model fit
            plt.scatter(self.x[:self.nlow, 0], self.x[:self.nlow, 1], color='black', marker='^', s=7.5, alpha=1.0)
            plt.scatter(self.x[self.nlow:self.ntot, 0], self.x[self.nlow:self.ntot, 1], color='red', marker='^', s=7.5, alpha=1.0)

            plt.colorbar(contour, label='$f(x,y)$')
            plt.grid(True)
            plt.title(rf"$\hat{{f}}(x,y)$ ($n={len(self.x)}$)")

        else:
            raise NotImplementedError

        plt.show()


    # todo error plot