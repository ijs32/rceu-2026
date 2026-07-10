import numpy as np

from shared.cpcrr.methods.cokrigingfitter import CoKrigingFitter
from shared.cpcrr.functions import onevar, onevarAD

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
y_c = onevarAD(x_c)
nlow = x_c.shape[0]

x_e = datax = np.array([
    0.0,
    0.24,
    0.5,
    0.65,
    # 0.82,
    # 0.9,
    1.0,
])
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


# CKF.plot_check_inputs(objective=onevar)


CKF.fit(
    # verbose=True,
)


print("Done.")
print(f"µ {CKF.globmean} σ2 {CKF.globvar}")
print(f"θ {CKF.theta}")
print(f"result {CKF.result}")

CKF.evaluate(1)