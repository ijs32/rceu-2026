from shared.cpcrr.methods import KrigingFitter
from shared.cpcrr.functions import branin

from shared.sampling import random_latin_hypercube, space_filling_latin_hypercube

size = 100

X_rlhc = random_latin_hypercube(size, 2)
datax,_,_  = space_filling_latin_hypercube(X_rlhc)

datay = branin(datax[:,0], datax[:,1])

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
print(f"result {KF.result}")


# KF.plot_check_model(objective=branin, res=20)

KF.test_check_model()



