

import numpy as np

import matplotlib.pyplot as plt

from scipy.linalg import solve_triangular
from scipy.optimize import minimize



class KrigingFitter:
    """
    State manager for Kriting fitting experiments.

    :param xmin: vector range (xmin, xmax)
    :param xmax: vector range (xmin, xmax)
    :param datax: array of x-values
    :param datay: 1d array of y-values

    """

    def __init__(self, xmin, xmax, datax, datay, ymin = None, ymax = None):
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
        self.n = self.x.shape[0]

        # global mean and variance
        # Note: global variance can also be called 'sill' or 'process variance'.
        self.globmean = np.zeros([1])
        self.globvar = np.zeros([1])
        self.theta = np.zeros([self.dim])
        # Psi, correlation matrix, and sqrt Psi
        self.Psi = np.zeros([self.n, self.n])
        self.sqrtPsi = None

        self.fig = plt.figure(figsize=(6, 5), dpi=100)

        # Scipy optimizer result
        self.result = None


    def fit(self, verbose = False):
        """
        For the given data set (x,y), perform a Kriging fit.

        :param verbose: verbosity
        """
        def negative_concentrated_log_likelihood(theta):
            # need to construct Psi in terms of the argument theta,
            # so that the minimizer can iteratively update theta.
            Psi = np.eye(self.n)*(1.0+1e-11)
            for i in range(0,self.n):
                for j in range(i+1, self.n):
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
            tmp = solve_triangular(self.sqrtPsi, self.y, lower=True)
            tmp2 = solve_triangular(self.sqrtPsi, np.ones([self.n]), lower=True)
            # print(f"tmp {tmp} tmp2 {tmp2}")
            self.globmean = np.dot(tmp, tmp2)
            self.globmean /= np.dot(tmp2, tmp2)
            if verbose:
                print(f"globmean {self.globmean} globvar {self.globvar}")
            # find global variance (needed for concentrated log likelihood)
            tmp = solve_triangular(self.sqrtPsi, self.y - self.globmean, lower=True)
            self.globvar = np.dot(tmp, tmp)/self.n
            # find concentrated log likelihood
            LnDetPsi = 2*np.sum(np.log(np.abs(np.diagonal(self.sqrtPsi))))
            return self.n/2 * np.log(self.globvar) + 0.5 * LnDetPsi

        # initialize
        thetamin = 1e-3
        thetamax = 1e2
        theta0 = np.full([self.dim], 0.1)
        # find theta using a numerical method
        result = minimize(
            negative_concentrated_log_likelihood,
            theta0,
            bounds=self.dim*[(thetamin,thetamax)],
            method='L-BFGS-B',
        )
        if verbose:
            print(f"theta {result.x}")
            print("FULL RESULT", result)
        # re-run to lock in final state after theta is found
        negative_concentrated_log_likelihood(result.x)
        self.theta = result.x
        # for diagnostics
        self.result = result

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
        psi = np.zeros([self.n])
        for i in range(self.n):
            prod = (self.x[i,:] - x)**2
            prod = np.dot(self.theta, prod)
            cij = np.exp(-prod)
            psi[i] = cij

        y = self.globmean
        tmp = solve_triangular(self.sqrtPsi, psi, lower=True)
        tmp2 = solve_triangular(self.sqrtPsi, self.y - self.globmean, lower=True)
        y += np.dot(tmp, tmp2)
        return y

    def evaluate_uncertainty(self, x):
        """
        Evaluate and also find the uncertainty at the same point.
        :param x:
        :return: y, u
        """
        # build psi
        psi = np.zeros([self.n])
        for i in range(self.n):
            prod = (self.x[i,:] - x[0])**2
            prod = np.dot(self.theta, prod)
            cij = np.exp(-prod)
            psi[i] = cij

        y = self.globmean
        tmp = solve_triangular(self.sqrtPsi, psi, lower=True)
        tmp2 = solve_triangular(self.sqrtPsi, self.y - self.globmean, lower=True)
        tmp3 = solve_triangular(self.sqrtPsi, np.ones(self.n), lower=True)
        y += np.dot(tmp, tmp2)
        term1 = np.dot(tmp, tmp)
        numer = 1.0 - np.dot(tmp3, tmp)
        denom = np.dot(tmp3, tmp3)
        sigma2 = self.globvar*(1.0 - term1 + numer/denom)
        u = np.sqrt(np.abs(sigma2))
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

    def plot_check_model(self, res = 100, objective=None):
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
                y[i], u[i] = self.evaluate_uncertainty(x[i].reshape(-1,1))
            plt.fill_between(
                x, y-u, y+u,
                color='#E0E4E8',
                alpha=0.6,
                edgecolor='none',
                label=r'$\pm\sigma$',
            ),
            plt.plot(
                x, y,
                color='black',
                alpha=0.6,
                linewidth=0.9,
                linestyle='-',
                label=r'$\hat{y}$',
                zorder=3,
            )
            plt.plot(
                x, objective(x),
                color='blue',
                alpha=0.6,
                linewidth=0.9,
                linestyle='--',
                label='$y$',
                zorder=3,
            )
            scatter_config1 = {
                'color':'black',
                'marker':'^',
                's':10,
                'alpha':1.0,
                'clip_on':False,
            }
            scatter_config2 = {
                'color':'red',
                's':20,
                # 'edgecolors':'#ffffff',
                'linewidths':1.1,
                'zorder':4,
                'label':'$X,Y$',
                'clip_on':False,
            }
            plt.scatter(
                self.x, self.y,
                **scatter_config2,
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
            plt.title(f"Kriging $\hat{{y}}$ ($n={len(self.x)}$)")
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
            plt.scatter(self.x[:, 0], self.x[:, 1], color='black', marker='^', s=7.5, alpha=1.0)
            plt.colorbar(contour, label='$f(x,y)$')
            plt.grid(True)
            plt.title(f"$\hat{{f}}(x,y)$ ($n={len(self.x)}$)")

        else:
            raise NotImplementedError

        plt.show()


    # todo error plot




