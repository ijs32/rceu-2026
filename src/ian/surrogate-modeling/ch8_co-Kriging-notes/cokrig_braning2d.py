import numpy as np

from shared.cpcrr.methods import KrigingFitter
from shared.sample_funcs import branin, braninAD

from shared.sampling import random_latin_hypercube, space_filling_latin_hypercube, find_subset, align_subset

nlow = 50
nhigh = 40
k = 2

X_rlhc = random_latin_hypercube(nlow, k, [-3,-3],[3,3])
X_sample,q,_ = space_filling_latin_hypercube(X_rlhc)

subset = find_subset(X_sample, nhigh, q)
datax  = align_subset(X_sample,subset)

x_c = datax[:nlow]
x_e = datax[:nhigh]

y_c = braninAD(x_e[:,0],x_e[:,1])
y_e = branin(x_e[:,0],x_e[:,1])

datay = np.concatenate([y_c, y_e])

KF = KrigingFitter(
    xmin=(0,0),
    xmax=(1,1),
    datax=datax,
    datay=datay,
)


# KF.plot_check_inputs(objective=branin)


KF.fit(
    # verbose=True,
)


print("Done.")
print(f"µ {KF.globmean} σ2 {KF.globvar}")
print(f"θ {KF.theta}")
# print(f"result {KF.result}")


# KF.plot_check_model(objective=branin, res=20)

KF.test_check_model()



