





from cpcrr.functions import onevar, branin

from cpcrr.experiment import Twokr

import numpy as np

from cpcrr.methods import KrigingFitter



target = branin
# target = onevar

delta = 0.1

# number of sample point to build surrogate model
N = 5

xs = [
    # 0.0,
    0.1,
    # 0.2,
    # 0.3,
    # 0.4,
    # 0.5,
    # 0.6,
    # 0.7,
    # 0.8,
    # 0.9,
]
ys = [
    # 0.0,
    0.1,
    # 0.2,
    # 0.3,
    # 0.4,
    # 0.5,
    # 0.6,
    # 0.7,
    # 0.8,
    # 0.9,
]


if __name__ == '__main__':

    for x in xs:
        for y in ys:
            xmin, xmax = x-delta, x+delta
            ymin, ymax = y-delta, y+delta
            x_sample = np.random.uniform(xmin, xmax, N)
            y_sample = np.random.uniform(ymin, ymax, N)
            datax = np.column_stack((x_sample,  y_sample))
            datay = branin(x_sample, y_sample)
            model = KrigingFitter(
                xmin=(0,0),
                xmax=(1,1),
                datax=datax,
                datay=datay,
            )
            model.fit(
                # verbose=True,
            )

            specification = ""
            response = "z\n"
            specification += f'x, {x-delta}, {x+delta}\n'
            specification += f'y, {y-delta}, {y+delta}\n'
            response += f'{model(x-delta, y-delta)}\n'
            response += f'{model(x-delta, y+delta)}\n'
            response += f'{model(x+delta, y-delta)}\n'
            response += f'{model(x+delta, y+delta)}\n'
            twokr = Twokr(
                specification=specification,
                verbose=True,
            )
            print(f"N {N}")
            print(f"x {x} y {y}")
            twokr.start(response=response)
            # todo multiple runs: logs are distinguishable :P







