import numpy as np

def random_latin_hypercube(n:int, k:int) -> np.ndarray:

    rng = np.random.default_rng()

    X = np.zeros((n,k))

    for i in range(k):
        X[:,i] = rng.permutation(n)

    return X