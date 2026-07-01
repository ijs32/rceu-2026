



from cpcrr.functions import onevar, branin

from cpcrr.experiment import Twokr

import numpy as np

from cpcrr.methods import KrigingFitter


target = onevar

delta = 0.1

# number of sample point to build surrogate model
N = 3

xs = [
    0.0,
    # 0.1,
    # 0.2,
    # 0.3,
    # 0.4,
    # 0.5,
    # 0.6,
    # 0.7,
    # 0.8,
    # 0.9,
    # 1.0,
]

if __name__ == '__main__':

    for x in xs:
        xmin, xmax = x-delta, x+delta
        x_sample = np.random.uniform(xmin, xmax, N)
        datax = x_sample
        datay = target(x_sample)
        model = KrigingFitter(
            xmin=(0,),
            xmax=(1,),
            datax=datax,
            datay=datay,
        )
        model.fit(
            # verbose=True,
        )


        specification = ""
        response = "y\n"
        specification += f'x, {x-delta}, {x+delta}\n'
        response += f'{model(x-delta)}\n'
        response += f'{model(x+delta)}\n'
        twokr = Twokr(
            specification=specification,
            # verbose=True,
        )
        twokr.start(response=response)


