import numpy as np

def random_latin_hypercube(
        n:int, 
        k:int, 
        edges:int = 0,
        lb: int = 0,
        ub: int = 1
    ) -> np.ndarray:

    rng = np.random.default_rng()

    X = np.zeros((n,k))

    for i in range(k):
        X[:,i] = rng.permutation(n)

    print(X)
    
    if edges == 1:
        X = (X-1)/(n-1)
    else:
        X = (X-0.5)/n

    return X