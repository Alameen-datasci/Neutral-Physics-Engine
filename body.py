"""
Defines the Body class.

A Body represents a point mass with position and velocity
used in the physics simulation.
"""

import numpy as np

class Body:
    """
    Represents a physical body in the simulation.

    Attributes:
        mass : float
            Mass of the body
        pos : np.ndarray
            Current position vector
        vel : np.ndarray
            Current velocity vector
    """
    def __init__(self, mass, pos, vel):
        self.mass = mass
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)