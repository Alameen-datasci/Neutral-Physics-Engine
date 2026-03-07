"""
forces.py

Defines the forces function which computes the gravitational accelerations on a list of Body objects.

The forces function calculates the net gravitational acceleration on each body due to all other bodies in the system using Newton's law of universal gravitation. It returns an array of acceleration vectors corresponding to each body.
"""

import numpy as np

G = 6.67430e-11
EPS = 1e-10

def forces(masses, pos):
    """
    Compute the gravitational accelerations on each body due to all other bodies.

    Parameters:
    -----------
    masses : list of float
        List of masses for each body in the simulation.
    pos : np.ndarray
        Array of shape (n, 3) containing the positions of each body.

    Returns:
    --------
    np.ndarray
        Array of shape (n, 3) containing the acceleration vectors for each body.
    
    The function iterates over all unique pairs of bodies, computes the gravitational force between them, and updates the acceleration for each body accordingly.
    It also includes a check to avoid singularities when bodies are very close to each other by skipping the force calculation if the distance is below a small threshold (EPS).
    """
    n = len(masses)
    accs = np.zeros((n, 3), dtype=float)

    for i in range(n):
        for j in range(i+1, n):
            r = pos[i] - pos[j]
            d = np.linalg.norm(r)
            # Avoid singularity and extremely large forces when bodies are very close
            if d < EPS:
                continue
            # Compute the common factor for the gravitational force
            f_common = G / (d ** 3)

            accs[i] -= f_common * masses[j] * r # Acceleration on body i due to body j
            accs[j] += f_common * masses[i] * r # Acceleration on body j due to body i (equal and opposite)
    return accs