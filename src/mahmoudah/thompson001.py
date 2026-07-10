

import numpy as np
from scipy.optimize import basinhopping
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D


def get_high_dim_reflection_matrix(p1):
    """
    Returns a d x d Householder matrix H that maps the unit vector p1
    exactly to the target vector [1, 0, 0, ..., 0].
    """
    d = len(p1)
    # Define our target vector t = [1, 0, 0, ..., 0]
    target = np.zeros(d)
    target[0] = 1.0
    # If p1 is already exceptionally close to the target, return Identity
    if np.allclose(p1, target):
        return np.eye(d)
    # If p1 is exactly opposite to the target [-1, 0, 0, ...],
    # a simple sign flip on the first coordinate handles it
    if np.allclose(p1, -target):
        H = np.eye(d)
        H[0, 0] = -1.0
        return H
    # 1. Compute the vector pointing from target to p1
    v = p1 - target
    # 2. Normalize it
    u = v / np.linalg.norm(v)
    # 3. Compute H = I - 2 * outer(u, u)
    # np.outer(u, u) creates the d x d matrix
    H = np.eye(d) - 2.0 * np.outer(u, u)
    return H



def rodriguez_formula(p1):
    """
    Compute the rotation matrix using Rodrigues' rotation formula,
    for 3D rotations.
    Reference: https://en.wikipedia.org/wiki/Rodrigues%27_rotation_formula
    """
    target = np.array([1.0, 0.0, 0.0])
    v = np.cross(p1, target)
    s = np.linalg.norm(v)
    c = np.dot(p1, target)
    if s < 1e-10:
        # If p1 is already at (1,0,0) or (-1,0,0)
        if c < 0:
            # It's exactly at (-1,0,0), rotate 180 degrees around Y axis
            R = np.array([[-1, 0, 0], [0, 1, 0], [0, 0, -1]])
        else:
            # It's already at (1,0,0), identity matrix (no rotation needed)
            R = np.eye(3)
    else:
        # Skew-symmetric cross-product matrix of v
        vx = np.array([[0, -v[2], v[1]],
                       [v[2], 0, -v[0]],
                       [-v[1], v[0], 0]])
        R = np.eye(3) + vx + np.dot(vx, vx) * ((1 - c) / (s ** 2))
    return R


def solve(n_points, dimension):

    def energy(flat_coords):
        """
        Calculate the total electrostatic potential energy of the system.
        """
        # Reshape flat array back to (N, d)
        points = flat_coords.reshape((n_points, dimension))
        # Project points to ensure they stay on the unit sphere during optimization
        norms = np.linalg.norm(points, axis=1, keepdims=True)
        points = points / norms
        energy = 0.0
        # Pairwise energy calculation: sum(1 / distance)
        for i in range(n_points):
            for j in range(i + 1, n_points):
                dist = np.linalg.norm(points[i] - points[j])
                # Avoid division by zero just in case
                energy += 1.0 / max(dist, 1e-10)
        return energy

    # Initialize points randomly in d-space and normalize to the unit sphere
    np.random.seed(42)  # For reproducible results
    initial_points = np.random.normal(size=(n_points, dimension))
    initial_points /= np.linalg.norm(initial_points, axis=1, keepdims=True)
    flat_initial = initial_points.flatten()
    # We use Scipy Basinhopping to escape local minima traps
    # 'SLSQP' is a local minimizer that handles our continuous space well
    minimizer_kwargs = {"method": "SLSQP"}
    res = basinhopping(
        energy,
        flat_initial,
        minimizer_kwargs=minimizer_kwargs,
        niter=50  # Increase iterations for higher N to get a better global minimum
    )
    # Extract and project final optimal coordinates
    coords = res.x.reshape((n_points, dimension))
    coords /= np.linalg.norm(coords, axis=1, keepdims=True)
    print(f"Minimum Energy Found: {res.fun:.4f}")
    # Define our source vector (the 1st point) and target vector
    p1 = coords[0]
    print(f"Rotating {p1} to (1,0,0,...,0)")
    R = get_high_dim_reflection_matrix(p1)
    # Rotate all points by multiplying by the rotation matrix R
    # final_coords is (N, 3), so we multiply by R.T to do: points * R^T (equivalent to R * vector)
    rotated_coords = np.dot(coords, R.T)
    return rotated_coords




def emit_points(coords):
    print(coords)






def visualize(coords):
    if coords.shape[1] != 3:
        raise ValueError("Visualization routine assumes 3d")
    fig = plt.figure(figsize=(8, 8))
    ax = fig.add_subplot(111, projection='3d')
    # Draw a wireframe sphere for visual reference
    u = np.linspace(0, 2 * np.pi, 30)
    v = np.linspace(0, np.pi, 30)
    x_sphere = np.outer(np.cos(u), np.sin(v))
    y_sphere = np.outer(np.sin(u), np.sin(v))
    z_sphere = np.outer(np.ones(np.size(u)), np.cos(v))
    ax.plot_wireframe(x_sphere, y_sphere, z_sphere, color='gray', alpha=0.1, linewidth=0.5)
    # Plot the optimized points
    ax.scatter(
        coords[:, 0],
        coords[:, 1],
        coords[:, 2],
        color='crimson',
        s=100,
        edgecolor='black',
        zorder=5,
        label='Electrons'
    )
    # Formatting the 3D plot
    ax.set_title(f"Thomson Problem Solution (N={n_points})", fontsize=14, fontweight='bold')
    ax.set_xlim([-1.1, 1.1])
    ax.set_ylim([-1.1, 1.1])
    ax.get_proj()
    ax.set_zlim([-1.1, 1.1])
    # ax.axis('off')
    plt.legend(loc="upper left")
    plt.show()



if __name__ == '__main__':

    n_points = 64
    dimension = 4

    print(f"Optimizing {n_points} points on a {dimension}D sphere...")

    coords = solve(n_points=n_points, dimension=dimension)

    print("Done.")

    emit_points(coords)
    if dimension == 3:
        visualize(coords)
