import numpy as np

def random_latin_hypercube(
        n:int, 
        k:int,
        lbs: list = [],
        ubs: list = []
    ) -> np.ndarray:
    """
    Generate a random Latin hypercube sampling plan in k dimensions.

    Each dimension is partitioned into n equal-width bins, and the plan
    places exactly one point per bin in every dimension (the Latin
    hypercube property), giving more uniform coverage of the domain than
    independent uniform sampling. Points are positioned at bin centres and
    then scaled from the unit hypercube [0, 1]^k onto [lb, ub]^k.

    Parameters
    ----------
    n : int
        Number of sample points. Also the number of bins per dimension,
        since the plan places one point per bin.
    k : int
        Number of dimensions (design variables).
    lbs : list, optional
        List of lower bound of the domain in every dimension (default 0).
    ubs : list, optional
        List of upper bound of the domain in every dimension (default 1).
        Must satisfy ub > lb.

    Returns
    -------
    np.ndarray
        An (n, k) array of sample points. Each column is a permutation of
        the same n bin centres, mapped to [lb, ub].

    Raises
    ------
    ValueError
        If len(ub) != len(lb).
    """
    if ubs == [] or lbs == []:
        lbs = [0]*k
        ubs = [1]*k

    if not len(ubs) == len(lbs) == k:
        raise ValueError("Upper and Lower bounds do not match eachother or k")

    for ub, lb in zip(ubs, lbs):
        if ub <= lb:
            raise ValueError("upper bound must be greater than lower bound")
    
    rng = np.random.default_rng()

    X = np.zeros((n,k))

    for i in range(k):
        X[:,i] = rng.permutation(n)
    
    for i, (ub, lb) in enumerate(zip(ubs, lbs)):
        X[:,i] = (X[:,i] + 0.5)/n
        w = ub - lb

        X_scaled = lb + w*X

    return X_scaled