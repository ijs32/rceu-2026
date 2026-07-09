__all__ = [
    "function_noise_1d",
    "onevar",
    "onevarAD",
    "aerofoil_drag",
    "branin",
    "liftsurf",
    "liftsurf_4d",
    "liftsurf_norm",
    "liftsurf_4d_norm",
    #
    "quickplot_hist",
    "quickplot_1d",
    "quickplot_scatter",
]

from .functions import (
    function_noise_1d,
    onevar,
    onevarAD,
    aerofoil_drag,
    branin,
    liftsurf,
    liftsurf_4d,
    liftsurf_norm,
    liftsurf_4d_norm,
)

from .functions_quickplot import (
    quickplot_hist,
    quickplot_1d,
    quickplot_scatter,
)


