

import numpy as np


def full_factorial(q, location = "edges"):
    """
    A full factorial sampling plan in the unit cube.


    :param q: 1-d integer array, describing the per-dimension sample sizes
    :param location: string, either "edges" or "centers".
    :return: full factorial sampling plan
    """
    # number of points in the sampling plan
    n = np.prod(q)
    # number of dimensions
    num_dim = len(q)

    X = np.zeros((n,num_dim))


    for j in range(num_dim):
        if location == "edges":
            one_d_slice = 123
        else:
            one_d_slice = 234

        # ...........................





