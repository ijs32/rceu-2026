

import numpy as np
from numpy.polynomial import Polynomial

savefig = False

import matplotlib.pyplot as plt


class Polynomial1DFitter:
    """
    State manager for 1D polynomial fitting experiments.

    :param xmin: scalar range (xmin, xmax)
    :param xmax: scalar range (xmin, xmax)
    :param datax: 1d array of x-values
    :param datay: 1d array of y-values
    :param tag: string for artifact storing
    """

    def __init__(self, xmin, xmax, datax, datay, tag = 'poly', ymin = None, ymax = None):
        plt.figure(figsize=(6, 5))
        self.xmin = xmin
        self.xmax = xmax
        M, m = datay.max(), datay.min()
        extent = (M-m)
        self.ymin = ymin if ymin is not None else m-0.1*extent
        self.ymax = ymax if ymax is not None else M+0.1*extent
        self.x = datax
        self.y = datay

        self.p = None

        self.tag = tag
        self.plot_config = {
            # Darker, muted hex color instead of default neon blue
            'color': '#1F3A52',
            # Thickens the line slightly to make it authoritative
            'linewidth': 0.7,
            # Options: '-', '--', ':', '-.'
            'linestyle': '-',
            # Adds shapes to data points ('o', 's', '^', 'D')
            # 'marker': 'o',
            # Size of the point markers
            # 'markersize': 6,
            # Hollows out the marker, making it clean and minimal
            # 'markerfacecolor': 'white',
            # Matches the marker border to the line color
            # 'markeredgecolor': '#1F3A52',
            # Thickness of the marker's border
            # 'markeredgewidth': 1.5,
        }
        self.scatter_config = {
            # Size of points: slightly smaller points look crisper
            's': 42,
            # Color: Avoid raw 'b' or 'blue'. Use a clean hex code.
            'c': '#2b5c8f',
            # Transparency: Crucial for showing data density where points overlap.
            'alpha': 0.6,
            # White borders make overlapping points "pop" and look distinct.
            'edgecolors': 'white',
            # Keep the border thin so it doesn't overpower the point.
            'linewidths': 0.5,
        }

        self.fig = plt.figure(figsize=(6, 5))


    def fit(self, order, plot = False, title = None, fname = None):
        """
        For the given data set (x,y), perform a polynomial
        fit and optionally produce a plot.

        :param order: order
        :param plot: optional, generate a plot showing the fit with the data.
        :param title: optional title
        :param fname: optional filename (along with tag)
        :return: numpy.Polynomial
        """
        p = Polynomial.fit(self.x, self.y, order)
        if plot:
            xplot = np.linspace(0,1, 200)
            # the curve of the fitting polynomial
            yplot = p(xplot)
            ax = self.fig.add_subplot(1,1,1)
            ax.plot(xplot, yplot, **self.plot_config)
            if title:
                ax.set_title(title)
            else:
                ax.set_title(f"Order {order}")
            ax.set_xlabel('$x$')
            ax.set_ylabel('$y$')
            ax.set_xlim((self.xmin, self.xmax))
            ax.set_ylim((self.ymin, self.ymax))
            ax.scatter(self.x, self.y, **self.scatter_config)
            self.fig.tight_layout()
            if savefig:
                if fname is not None:
                    self.fig.savefig(f"{self.tag}.{fname}.png")
                else:
                    self.fig.savefig(f"{self.tag}.order{order}.png")
                self.fig.clear()
        self.p = p
        return p



    def fit_and_cross_validate(
            self,
            order,
            n_folds = 5,
            plot = False,
            seed = None,
    ):
        """
        For the given number of folds $k$, perform standard $k$-fold cross-validation
        with the dataset, on the model completely determined by the single
        parameter `order`.

        Set the seed to a fixed integer if reproducibility is needed.
        Or, if comparing cross-validation across different values of `order`,
        set the seed in order to ensure that the folds are the same across
        different orders. 

        :param order: the order of the 1d-polynomial-fitting model. (e.g., order = 3, cubic polynomial)
        :param n_folds:
        :param plot:
        :param seed: positive integer
        :return: loss, mean, stddev: 1d array of losses, mean of `loss`, stddev of `loss`
        """
        N = len(self.x)
        k = n_folds
        s, remainder = divmod(N, k)
        if remainder > 0:
            print(f"Fold value {k} does not evenly divide sample size {N}. Stopping early.")
            return
        # Have s, the fold size.
        rng = np.random.default_rng(seed=seed)
        loss = np.zeros(k)
        permutation = rng.permutation(len(self.x))
        x_original, y_original = self.x, self.y
        x, y = self.x[permutation], self.y[permutation]
        for i in range(k):
            # ith fold
            self.x = np.hstack((x[0:s*i], x[s*(i+1):]))
            self.y = np.hstack((y[0:s*i], y[s*(i+1):]))
            title = f"order {order}, fold {i}"
            fname = f"CV.fold{i}order{order}"
            model = self.fit(order, plot, title, fname)
            testx = x[s*i:s*(i+1)]
            testy = y[s*i:s*(i+1)]
            modely = model(testx)
            loss[i] = np.sqrt(((testy - modely)**2).mean())
        mean = loss.mean()
        stddev = loss.std()
        self.x, self.y = x_original, y_original
        return loss, mean, stddev


    def evaluate(self, x):
        """
        Evaluate the model.

        :param x:
        """
        y = self.p(x)
        return y

