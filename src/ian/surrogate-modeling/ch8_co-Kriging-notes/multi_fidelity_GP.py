import numpy as np

from shared.cpcrr.functions import onevar, onevarAD
from shared.surrogates.emukitMFGP import EmukitMFGP
from shared.plotter import Plotter

x_c = np.array([
    0,
    0.24,
    0.5,
    0.65,
    1.0,
    0.1,
    0.2,
    0.3,
    0.5,
    0.7,
    0.8,
    0.9
])
nlow = x_c.shape[0]

x_e = np.array([
    0.0,
    0.24,
    0.5,
    0.65,
    # 0.82,
    # 0.9,
    1.0,
])

datax = np.concatenate([x_c, x_e])
if len(datax.shape) == 1:
    datax = datax.reshape(-1,1)
    
gp = EmukitMFGP(datax, onevar, onevarAD, nlow)
plotter = Plotter(
    gp,
    xmin=(0,),
    xmax=(1,),
)

plotter.plot_check_model()