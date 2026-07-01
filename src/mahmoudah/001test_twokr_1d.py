



from cpcrr.functions import onevar, branin

from cpcrr.experiment import Twokr

import numpy as np


target = onevar

delta = 0.1

xs = [
    # 0.0,
    # 0.1,
    0.2,
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

        specification = ""
        response = "y\n"
        specification += f'x, {x-delta}, {x+delta}\n'
        response += f'{target(x-delta)}\n'
        response += f'{target(x+delta)}\n'
        twokr = Twokr(
            specification=specification,
            # verbose=True,
        )
        twokr.start(response=response)


