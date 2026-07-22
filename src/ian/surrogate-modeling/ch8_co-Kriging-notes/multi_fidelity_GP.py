from shared.sample_funcs import peaks, peaksAD
from shared.surrogates.emukitMFGP import EmukitMFGP
from shared.plotter import Plotter
from shared.sampling import random_latin_hypercube, space_filling_latin_hypercube, find_subset, align_subset

nlow = 50
nhigh = 40
k = 2

X_rlhc = random_latin_hypercube(nlow, k, [-3,-3],[3,3])
X_sample,q,_ = space_filling_latin_hypercube(X_rlhc)

subset = find_subset(X_sample, nhigh, q)
datax  = align_subset(X_sample,subset)

xmin=(-3,-3)
xmax=(3,3)

gp = EmukitMFGP(datax, peaks, peaksAD, nlow, xmin, xmax)
print(gp.gpy_linear_mf_model)

plotter = Plotter(gp)

plotter.plot_check_model()