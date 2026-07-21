import numpy as np

from shared.cpcrr.methods.cokrigingfitter import CoKrigingFitter
from shared.sample_funcs import peaks, peaksAD
from shared.sample_funcs import peaks, peaksAD

from shared.sampling import random_latin_hypercube, space_filling_latin_hypercube, find_subset, align_subset

sample_sizes = [i*5 for i in range(1,5)]
# globmeans, globvars, thetas, thetads, rhos, psis, min_dists, LnDetPsis, Sigmas = [], [], [], [], [], [], [], [], []

for size in sample_sizes:
    nlow = size*2
    nhigh = size
    k = 2

    X_rlhc = random_latin_hypercube(nlow, k, [-3]*2,[3]*2)
    x,q,_ = space_filling_latin_hypercube(X_rlhc)

    subset = find_subset(x, nhigh, q)
    datax  = align_subset(x,subset)

    x_c = datax[:nlow]
    x_e = datax[:nhigh]

    y_c = peaksAD(x_c[:,0],x_c[:,1])
    y_e = peaks(x_e[:,0],x_e[:,0])

    datay = np.concatenate([y_c, y_e])

    CKF = CoKrigingFitter(
        xmin=(-3,-3),
        xmax=(3,3),
        datax=datax,
        datay=datay,
        nlow=nlow
    )

    CKF.fit(
        # verbose=True,
    )


    print("Done. ",size)
    print(CKF.test_check_model())
    # print(CKF.theta)
    # print(CKF.thetad)

#     globmeans.append(CKF.globmean)
#     globvars.append(CKF.globvar)
#     thetas.append(CKF.theta)
#     thetads.append(CKF.thetad)
#     rhos.append(CKF.rho)
#     psis.append(np.linalg.cond(CKF.sqrtPsi))
#     min_dists.append(scipy.spatial.distance.pdist(x_c).min())
#     LnDetPsis.append(CKF.LnDetPsi)
#     Sigmas.append(np.linalg.cond(CKF.sqrtSigma))


# variables = {
#     "globmean": globmeans,
#     "globvar": globvars,
#     "theta": thetas,
#     "thetad": thetads,
#     "rho": rhos,
#     "psis": psis,
#     "min_dist": min_dists,
#     "LnDetPsi": LnDetPsis,
#     "Sigmas": Sigmas
# }

# for name, values in variables.items():
#     plt.figure()
#     if name == "theta" or name == "thetad":
#         plt.plot(sample_sizes, values, "o-")
    
#     plt.title(name)
#     plt.xlabel("Sample Size")
#     plt.savefig(f"./test_outputs/2d/{name}.png", dpi=150)
#     plt.close()