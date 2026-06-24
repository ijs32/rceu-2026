import numpy as np

def psi(r: np.ndarray, code: int = 1, sigma: float = 0, theta: np.ndarray = np.ndarray) -> np.ndarray:
    match code:
        case 1: # linear
            return r
        case 2: # cubic
            return r**3
        case 3:  # thin plate spline
            # claude explained boolean indexing in numpy, very cool
            result = np.zeros_like(r)
            nz = r > 0
            result[nz] = (r[nz]**2) * np.log(r[nz])
            return result
        case 4: # Gaussian
            return np.exp(-(r**2)/(2*(sigma**2)))
        case 5: # multi-quadratic
            return np.sqrt(r**2 + sigma**2)
        case 6: # inverse multi-quadratic
            return 1 / (np.sqrt(r**2 + sigma**2))
        case 7: # Kriging
            return np.exp(-(theta*r))
        
        case _:
            return r