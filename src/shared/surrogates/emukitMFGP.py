from collections.abc import Callable

import GPy
import numpy as np
from emukit.test_functions.forrester import forrester, forrester_low
from emukit.multi_fidelity.kernels import LinearMultiFidelityKernel
from emukit.multi_fidelity.models import GPyLinearMultiFidelityModel
from emukit.multi_fidelity.convert_lists_to_array import convert_xy_lists_to_arrays

from shared.surrogates.mf_surrogate import MFSurrogate


class EmukitMFGP(MFSurrogate):

    def __init__(self, x, func_e: Callable, func_c: Callable, nlow, verbose = False):
        super().__init__(x, nlow, func_e, func_c, verbose)

        #     self.x_c = np.atleast_2d(np.random.rand(12)).T
        #     self.x_e = np.atleast_2d(np.random.permutation(self.x_c)[:6])

        self.arrange_training_data()
        self.fit_model()


    def arrange_training_data(self):
        y_train_l = self.func_c(self.x_c)
        y_train_h = self.func_e(self.x_e)

        self.X_train, self.Y_train = convert_xy_lists_to_arrays([self.x_c, self.x_e], [y_train_l, y_train_h])


    def fit_model(self):
        num_fidelities = 2
        kernels = [GPy.kern.RBF(1), GPy.kern.RBF(1)]
        for k in kernels:
            k.lengthscale.constrain_bounded(1e-2, 1.0)   # domain is [0,1], don't let it blow past that
            k.variance.constrain_bounded(1e-3, 1e3) 
        
        linear_mf_kernel = LinearMultiFidelityKernel(kernels)
        self.gpy_linear_mf_model = GPyLinearMultiFidelityModel(self.X_train, self.Y_train, linear_mf_kernel, n_fidelities = num_fidelities)

        self.gpy_linear_mf_model.mixed_noise.Gaussian_noise.fix(1e-6)
        self.gpy_linear_mf_model.mixed_noise.Gaussian_noise_1.fix(1e-6)

        self.gpy_linear_mf_model.optimize_restarts(num_restarts=10)
        self.gpy_linear_mf_model.optimize()


    def evaluate(self, x):
        x0 = np.append(x, 1.0)[None, :] # have to append 1 so .predict() knows we want predictions on high-fidelity function
        mean, _ = self.gpy_linear_mf_model.predict(x0)

        return mean
    

    def evaluate_uncertainty(self, x):
        x0 = np.append(np.atleast_1d(x), 1.0)[None, :] # append a 1 so gpy knows we want high fidelity predictions.
        Y_metadata = {'output_index': x0[:, -1:].astype(int)} # tell model which noise parameter to use (0 in our case anyways)
        
        mean, var = self.gpy_linear_mf_model.predict(x0, Y_metadata=Y_metadata)

        return mean, np.sqrt(var)