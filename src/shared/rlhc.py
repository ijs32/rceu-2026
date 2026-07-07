import numpy as np
import math
from scipy.spatial.distance import pdist

def random_latin_hypercube(
        n:int, 
        k:int,
        lbs: list = [],
        ubs: list = []
    ) -> np.ndarray:
    """
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
        if ubs[i] <= lbs[i]
        
        if not len(ubs) == len(lbs) == k
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
    
    X_scaled = np.zeros((n,k))
    for i, (lb, ub) in enumerate(zip(lbs, ubs)):
        X[:,i] = (X[:,i] + 0.5)/n
        w = ub - lb

        X_scaled[:,i] = lb + w*X[:,i]

    return X_scaled

def dist(X: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # # My original approach:
    # distances = []
    # for i in range(len(X)-1):
    #     for j in range(1,len(X)):
    #         distances.append(np.linalg.norm(X[i,:] - X[j,:]))
    
    # # round because normalized to [0,1]
    # return np.unique(distances, return_counts=True) # returns distinct_d, J

    # claude suggested using pdists from scipy instead:
    sq_dists = pdist(X, metric='sqeuclidean')
    return np.unique(sq_dists, return_counts=True) # returns distinct_d, J


def mm_phiq(X,q=2):
    d,J = dist(X)

    # morris mitchell sampling plan quality criterion
    return sum(((J/(d**q))**(1/q)))


def perturb(X: np.ndarray, pert_n=1):
    n = X.shape[0]
    k = X.shape[1]

    for _ in range(pert_n):
        col = int(np.floor(np.random.random()*k))
        
        el1 = el2 = 1
        while el1==el2:
            el1 = int(np.floor(np.random.random()*n))
            el2 = int(np.floor(np.random.random()*n))

        buffer = X[el1,col]
        X[el1,col] = X[el2,col]
        X[el2,col] = buffer

    return X


def best_lhc_by_q(X_start, pop, iter, q = 2):
    X_best = X_start; 
    phi_best = mm_phiq(X_start, q)

    n = X_best.shape[0]

    leveloff = math.floor(0.85*iter)
    for it in range(1,iter+1):
        if it < leveloff:
            mutations = round(1+(0.5*n-1)*(leveloff-it)/(leveloff-1))
        else:
            mutations = 1
    
        X_improved   = X_best
        phi_improved = phi_best
        
        for _ in range(pop):
            X_try = perturb(X_best, mutations)
            phi_try = mm_phiq(X_try, q)
            
            if phi_try < phi_improved:
                X_improved = X_try
                phi_improved = phi_try
        
        if phi_improved < phi_best:
            X_best = X_improved
            phi_best = phi_improved

    return X_best, phi_best


def space_filling_latin_hypercube(
        X_start: np.ndarray,
        pop:int = 25,
        iter:int = 25
    ) -> dict:
    """
    Generate a random Latin hypercube sampling plan in k dimensions.

    Each dimension is partitioned into n equal-width bins, and the plan
    places exactly one point per bin in every dimension (the Latin
    hypercube property), giving more uniform coverage of the domain than
    independent uniform sampling. Points are positioned at bin centres and
    then scaled from the unit hypercube [0, 1]^k onto [lb, ub]^k.

    Parameters
    ----------
    X_start : np.ndarray
        A random latin hypercube generated via random_latin_hypercube
    pop : int, optional
        Controls the number of perturbations done to the random latin hypercube
    iter : int, optional
        Controls the number of mutations done during a given perturbation

    Increasing pop and iter will improve the sampling plan but increase time
    to compute.
    
    Returns
    -------
    np.ndarray
        An (n, k) array of sample points. Each column is a permutation of
        the same n bin centres.
    """
    qs = [1,2,5,10,20,50,100]

    X_tracker = {
        "q": "",
        "phi": math.inf,
        "X": ""
    }

    for q in qs:
        X_best, phi_best = best_lhc_by_q(X_start, pop, iter, q)

        if phi_best < X_tracker["phi"]:
            X_tracker["q"]   = q
            X_tracker["phi"] = phi_best
            X_tracker["X"]   = X_best
    
    return X_tracker["X"], X_tracker["q"]

