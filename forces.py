"""
forces.py

Defines the forces function which computes the gravitational accelerations on a list of Body objects.

The forces function calculates the net gravitational acceleration on each body due to all other bodies in the system using Newton's law of universal gravitation. It returns an array of acceleration vectors corresponding to each body.
"""

import numpy as np

G = 6.67430e-11
EPS = 1e-10

def forces(bodies):
    """
    Compute the gravitational accelerations on each body due to all other bodies.

    Parameters:
    -----------
    bodies : list of Body
        List of Body objects in the simulation

    Returns:
    --------
    list of np.ndarray
        List of acceleration vectors for each body, where each vector is a numpy array of shape (dim,)
    
    The function iterates over all unique pairs of bodies, computes the gravitational force between them, and accumulates the resulting accelerations for each body.
    It handles cases where bodies are very close to avoid singularities by using a small epsilon value.
    """
    n = len(bodies)
    dim = bodies[0].pos.shape[0]
    accs = np.zeros((n, dim), dtype=float)

    for i in range(n):
        for j in range(i+1, n):
            r = bodies[i].pos - bodies[j].pos
            d = np.linalg.norm(r)
            # Avoid singularity and extremely large forces when bodies are very close
            if d < EPS:
                continue
            # Compute the common factor for the gravitational force
            f_common = G / (d ** 3)

            accs[i] -= f_common * bodies[j].mass * r # Acceleration on body i due to body j
            accs[j] += f_common * bodies[i].mass * r # Acceleration on body j due to body i (equal and opposite)
    return accs