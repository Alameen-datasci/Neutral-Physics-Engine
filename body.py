"""
Defines the Body class.

A Body represents a point mass with position and velocity
used in the physics simulation.
"""

import numpy as np

class Body:
    def __init__(self, mass, pos, vel):
        self.mass = mass
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)