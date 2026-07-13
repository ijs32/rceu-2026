import numpy as np
import matplotlib.pyplot as plt
import scipy

from shared.cpcrr.methods.cokrigingfitter import CoKrigingFitter
from shared.cpcrr.functions.functions import branin, braninAD
from shared.sampling import random_latin_hypercube, space_filling_latin_hypercube, find_subset, align_subset

sample_sizes = [i*5 for i in range(1,15)]
globmeans, globvars, thetas, thetads, rhos, psis, min_dists, LnDetPsis = [], [], [], [], [], [], [], []

for size in sample_sizes:
    nlow = size*2
    nhigh = size
    k = 2

    X_rlhc = random_latin_hypercube(nlow, k)
    X_sample = space_filling_latin_hypercube(X_rlhc)

    subset = find_subset(X_sample["X"], nhigh, X_sample["q"])
    datax  = align_subset(X_sample["X"],subset)

    x_c = datax[:nlow]
    x_e = datax[:nhigh]

    y_c = braninAD(x_c[:,0], x_c[:,1])
    y_e = branin(x_e[:,0], x_e[:,1])

    datay = np.concatenate([y_c, y_e])

    CKF = CoKrigingFitter(
        xmin=(0,0),
        xmax=(1,1),
        datax=datax,
        datay=datay,
        nlow=nlow
    )

    CKF.fit(
        # verbose=True,
    )


    print("Done. ",size)
    globmeans.append(CKF.globmean)
    globvars.append(CKF.globvar)
    thetas.append(CKF.theta[0])
    thetads.append(CKF.thetad[0])
    rhos.append(CKF.rho)
    psis.append(np.linalg.cond(CKF.sqrtPsi))
    min_dists.append(scipy.spatial.distance.pdist(x_c).min())
    LnDetPsis.append(CKF.LnDetPsi)


variables = {
    "globmean": globmeans,
    "globvar": globvars,
    "theta": thetas,
    "thetad": thetads,
    "rho": rhos,
    "psis": psis,
    "min_dist": min_dists,
    "LnDetPsi": LnDetPsis
}

for name, values in variables.items():
    plt.figure()
    plt.plot(sample_sizes, values, "o-")
    plt.title(name)
    plt.xlabel("Sample Size")
    plt.savefig(f"./test_outputs/2d/{name}.png", dpi=150)
    plt.close()