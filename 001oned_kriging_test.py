

import numpy as np

from cpcrr.methods import KrigingFitter

from cpcrr.functions import onevar


datax = np.array([
    0.0,
    0.24,
    0.5,
    0.65,
    # 0.82,
    # 0.9,
    1.0,
])
datay = onevar(datax)

KF = KrigingFitter(
    xmin=(0,),
    xmax=(1,),
    datax=datax,
    datay=datay,
)

# KF.plot_check_inputs(objective=onevar)

KF.fit(
    # verbose=True,
)

print("Done.")
print(f"µ {KF.globmean} σ2 {KF.globvar}")
print(f"θ {KF.theta}")
print(f"result {KF.result}")

KF.plot_check_model(res=100, objective=onevar)

KF.test_check_model()



