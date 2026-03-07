"""
body.py

This module defines the Body class, which represents a physical body in the simulation. Each Body instance has properties such as mass, position, velocity, and radius.
The class includes validation to ensure that the properties are physically meaningful (e.g., positive mass and radius) and that the position and velocity are 3D vectors.
The Body class serves as a fundamental building block for the physics engine, allowing us to represent and manipulate individual bodies in the simulation.
"""

import numpy as np


class Body:
    """
    Represents a physical body in the simulation with properties such as mass, position, velocity, and radius.

    Parameters:
    -----------
    mass : float
        Mass of the body (must be positive and non-zero)
    pos : array-like
        Initial position of the body as a 3D vector (shape (3,))
    vel : array-like
        Initial velocity of the body as a 3D vector (shape (3,))
    radius : float
        Radius of the body (must be positive)
    orientation : array-like, optional
        Initial orientation of the body as a quaternion (shape (4,)), default is None (not implemented)
    angular_vel : array-like, optional
        Initial angular velocity of the body as a 3D vector (shape (3,)), default is None (not implemented)

    Note: The Body class currently focuses on translational properties (mass, position, velocity, radius) and includes placeholders for rotational properties (orientation and angular velocity) that can be implemented in future versions of the physics engine.
    """

    def __init__(self, mass, pos, vel, radius, orientation=None, angular_vel=None):

        # ------------------------- Scalars -------------------------
        if mass <= 0:
            raise ValueError("Body mass must be positive and non-zero")
        self.mass = float(mass)

        if radius <= 0:
            raise ValueError("Body radius must be positive")
        self.radius = float(radius)

        # ------------------------- Translational State -------------------------
        self.pos = np.asarray(pos, dtype=float)
        self.vel = np.asarray(vel, dtype=float)

        if self.pos.shape != (3,):
            raise ValueError("Position must be a 3D vector of shape (3,)")

        if self.vel.shape != (3,):
            raise ValueError("Velocity must be a 3D vector of shape (3,)")

        # ------------------------- Defaults (Rotational) -------------------------
        # if orientation is None:
        #     orientation = [1, 0, 0, 0]

        # if angular_vel is None:
        #     angular_vel = [0, 0, 0]

        # ------------------------- Rotational State -------------------------
        # self.orientation = np.asarray(orientation, dtype=float)
        # self.angular_vel = np.asarray(angular_vel, dtype=float)

        # if self.orientation.shape != (4,):
        #     raise ValueError(
        #         "Orientation must be a quaternion represented as a 4D vector of shape (4,)"
        #     )

        # if self.angular_vel.shape != (3,):
        #     raise ValueError("Angular velocity must be a 3D vector of shape (3,)")

        # norm = np.linalg.norm(self.orientation)
        # if norm < 1e-12:
        #     raise ValueError("Orientation quaternion must be non-zero")
        # self.orientation /= norm
