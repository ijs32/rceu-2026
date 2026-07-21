import matplotlib.pyplot as plt
import numpy as np
from shared.surrogates.surrogate import Surrogate
from shared.surrogates.mf_surrogate import MFSurrogate


class Plotter():
    def __init__(self, surrogate: Surrogate | MFSurrogate, xmin, xmax):
        self.xmin = xmin
        self.xmax = xmax

        self.surrogate = surrogate
        self.fig = plt.figure(figsize=(6, 5), dpi=100)
        

    def plot_check_inputs(self, res = 100):
        """
        Produce a plot to inspect model
        :param res: resolution (sample values per dimension)
        """
        if self.surrogate.dim == 1:

            Nside = res
            x = np.linspace(self.xmin[0], self.xmax[0], Nside)
            y = self.surrogate.func_e(x)

            plt.plot(x, y, color='black', linewidth=0.5)

            if isinstance(self.surrogate, MFSurrogate):
                plt.scatter(self.surrogate.x_e, self.surrogate.y_e, color='black', marker='^', s=7.5, alpha=1.0)
                plt.title(f"$f(x)$ ($n_e={len(self.surrogate.x_e)}$)")
            else:
                plt.scatter(self.surrogate.x, self.surrogate.y, color='black', marker='^', s=7.5, alpha=1.0)
                plt.title(f"$f(x)$ ($n={len(self.surrogate.x)}$)")

            plt.xlim((self.xmin[0], self.xmax[0]))

        elif self.surrogate.dim == 2:
            Nside = res
            x0 = np.linspace(self.xmin[0], self.xmax[0], Nside)
            y0 = np.linspace(self.xmin[1], self.xmax[1], Nside)
            X, Y = np.meshgrid(x0, y0)
            datax_plot = np.column_stack((X.reshape(-1), Y.reshape(-1)))
            Z = self.surrogate.func_e(datax_plot[:,0], datax_plot[:,1]).reshape((Nside, Nside))
            contour = plt.contourf(X, Y, Z, levels=24, cmap='viridis', alpha=0.85)

            if isinstance(self.surrogate, MFSurrogate):
                plt.scatter(self.surrogate.x_e[:, 0], self.surrogate.x_e[:, 1], color='black', marker='^', s=7.5, alpha=1.0)
                plt.title(f"$f(x,y)$ ($n_e={len(self.surrogate.x_e)}$)")
            else:
                plt.scatter(self.surrogate.x[:, 0], self.surrogate.x[:, 1], color='black', marker='^', s=7.5, alpha=1.0)
                plt.title(f"$f(x,y)$ ($n={len(self.surrogate.x)}$)")

            plt.colorbar(contour, label='$f(x,y)$')
            plt.grid(True)
        else:
            raise NotImplementedError
        plt.show()


    def plot_check_model(self, res = 100):
        """
        Produce a plot to inspect model
        :param res: resolution (sample values per dimension)
        """
        if self.surrogate.dim == 1:
            # for 1d case we also plot uncertainty
            Nside = res
            x = np.linspace(self.xmin[0], self.xmax[0], Nside)
            y = np.ones(Nside)
            u = np.ones(Nside)

            if hasattr(self.surrogate, 'evaluate_uncertainty'):
                for i in range(Nside):
                    y[i], u[i] = self.surrogate.evaluate_uncertainty(x[i])
                    
                plt.fill_between(
                    x, y-u, y+u,
                    color='#E0E4E8',
                    alpha=0.6,
                    edgecolor='none',
                    label=r'$\pm\sigma$',
                )

            else:
                for i in range(Nside):
                    y[i] = self.surrogate.__call__(x[i])

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
                x, self.surrogate.func_e(x),
                color='black',
                alpha=0.6,
                linewidth=0.9,
                linestyle='-',
                label='$y_e$',
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

            if isinstance(self.surrogate, MFSurrogate):
                plt.plot(
                    x, self.surrogate.func_c(x),
                    color='red',
                    alpha=0.6,
                    linewidth=0.9,
                    linestyle='--',
                    label='$y_c$',
                    zorder=3,
                )

                plt.scatter(
                    self.surrogate.x_e, self.surrogate.y_e,
                    **scatter_config1,
                )
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
                    self.surrogate.x_c, self.surrogate.y_c,
                    **scatter_config2,
                )
            else:
                plt.scatter(
                    self.x, self.y,
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
            plt.title(rf"{type(self.surrogate).__name__} $\hat{{y}}$ ($n={len(self.surrogate.x)}$)")
        elif self.surrogate.dim == 2:

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

            Nside = res
            x0 = np.linspace(self.xmin[0], self.xmax[0], Nside)
            y0 = np.linspace(self.xmin[1], self.xmax[1], Nside)
            X, Y = np.meshgrid(x0, y0)
            datax_plot = np.column_stack((X.reshape(-1), Y.reshape(-1)))
            matr = np.hstack((X.reshape((-1, 1)), Y.reshape((-1, 1))))
            Z_true = self.surrogate.func_e(matr).reshape((Nside,Nside))

            Z = np.ones([Nside*Nside])
            for i in range(datax_plot.shape[0]):
                Z[i] = self.surrogate.__call__(datax_plot[i,:])
            Z = Z.reshape((Nside, Nside))

            fill1 = ax1.contourf(X, Y, Z_true, levels=24, cmap='viridis')
            ax1.contour(X, Y, Z_true, levels=24, colors='black', linewidths=0.8)
            ax1.set_xlabel(r'$x$')
            ax1.set_ylabel(r'$y$')
            ax1.set_aspect('equal')
            fig.colorbar(fill1, ax=ax1, shrink=0.5, aspect=10)

            fill2 = ax2.contourf(X, Y, Z, levels=24, cmap='viridis', alpha=0.85)
            ax2.contour(X, Y, Z, levels=24, colors='black', linewidths=0.8)

            # plot collocation points used to perform model fit
            ax2.scatter(self.x[:self.nlow, 0], self.x[:self.nlow, 1], color='black', marker='^', s=7.5, alpha=1.0)
            ax2.scatter(self.x[self.nlow:self.ntot, 0], self.x[self.nlow:self.ntot, 1], color='red', marker='^', s=7.5, alpha=1.0)

            ax2.set_xlabel(r'$x$')
            ax2.set_ylabel(r'$y$')
            ax2.set_aspect('equal')
            fig.colorbar(fill2, ax=ax2, shrink=0.5, aspect=10)


        else:
            raise NotImplementedError

        plt.show()


    # todo error plot