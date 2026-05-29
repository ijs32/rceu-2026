__all__ = [
    "function_noise_1d",
    "onevar",
    "onevarAD",
    "aerofoil_drag",
    "branin",
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
)

from .functions_quickplot import (
    quickplot_hist,
    quickplot_1d,
    quickplot_scatter,
)


