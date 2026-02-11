"""
Defines the Body class.

A Body represents a point mass with position, velocity and radius
used in the physics simulation.
"""

import numpy as np

class Body:
    """
    Represents a physical body in the simulation.

    Parameters:
        mass : float (kilogram)
            Mass of the body
        pos : np.ndarray
            Current position vector
        vel : np.ndarray
            Current velocity vector
        radius : float (meter)
            Radius of the body
    """
    def __init__(self, mass, pos, vel, radius):
        if mass <= 0:
            raise ValueError("Body mass must be positive and non-zero")
        self.mass = mass

        if radius <= 0:
            raise ValueError("Body radius must be positive")
        self.radius = radius

        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)

class Earth(Body):
    """
    Represents the Earth as a Physical Body in the simulation.

    This class inherits from Body and initializes with Earth's standard mass and radius.
    The initial position and velocity can be set, but default to zero (stationary at origin).

    Parameters:
    -----------
    pos : np.ndarray
        Initial position vector (default: [0, 0, 0])
    vel : np.ndarray
        Initial velocity vector (default: [0, 0, 0])

    Notes:
    ------
    Mass and radius are set to Earth's standard values:
    - Mass: 5.9722e24 kg
    - Radius: 6371008.8 m
    """
    MASS = 5.9722e24
    RADIUS = 6371008.8

    def __init__(self, pos=None, vel=None):
        if pos is None:
            pos = np.zeros(3)
        if vel is None:
            vel = np.zeros(3)

        super().__init__(
            mass=Earth.MASS,
            pos=pos,
            vel=vel,
            radius=Earth.RADIUS
        )