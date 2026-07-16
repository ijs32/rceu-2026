import numpy as np
import matplotlib.pyplot as plt

from shared.cpcrr.methods.cokrigingfitter import CoKrigingFitter
from shared.cpcrr.functions import onevar, onevarAD

sample_sizes = [i*5 for i in range(1,10)]
globmeans, globvars, thetas, thetads, rhos, psis = [], [], [], [], [], []

for size in sample_sizes:
    k = 2

    x_e = np.random.rand(size)
    x_c = np.concatenate([x_e, np.random.rand(size)])

    y_c = onevarAD(x_c)
    nlow = x_c.shape[0]

    y_e = onevar(x_e)

    datax = np.concatenate([x_c, x_e])
    datay = np.concatenate([y_c, y_e])

    if len(datax.shape) == 1:
        datax = datax.reshape(-1,1)

    CKF = CoKrigingFitter(
        xmin=(0,),
        xmax=(1,),
        datax=datax,
        datay=datay,
        nlow=nlow
    )

    CKF.fit(
        # verbose=True,
    )


    print("Done.")
    globmeans.append(CKF.globmean)
    globvars.append(CKF.globvar)
    thetas.append(CKF.theta)
    thetads.append(CKF.thetad)
    rhos.append(CKF.rho)
    psis.append(np.linalg.cond(CKF.sqrtPsi))


variables = {
    "globmean": globmeans,
    "globvar": globvars,
    "theta": thetas,
    "thetad": thetads,
    "rho": rhos,
}

for name, values in variables.items():
    plt.figure()
    plt.plot(sample_sizes, values, "o-")
    plt.title(name)
    plt.xlabel("Sample Size")
    plt.savefig(f"./test_outputs/1d/{name}.png", dpi=150)
    plt.close()