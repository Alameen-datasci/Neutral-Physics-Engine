"""
body.py

This module defines the Body class, the foundational physical entity for the
simulation engine. It encapsulates the core translational properties of a
macroscopic object—mass, spatial position, velocity, and collision radius—
while ensuring strict validation for physically meaningful parameters.

The Body class is designed to seamlessly integrate with spatial partitioning
structures and field evaluators. It establishes a robust framework for 3D
point-mass dynamics and includes architectural placeholders for future expansion
into rigid-body rotational mechanics (quaternion-based orientation and angular velocity).
"""

import numpy as np


class Body:
    """
    Represents a fundamental physical body within the simulation environment.

    The Body class manages the physical state of a single entity. It enforces
    strict dimensional and value constraints upon initialization to guarantee
    that the physics solver operates on mathematically and physically sound data.
    """

    def __init__(
        self,
        mass: np.float64,
        pos: list[float] | np.ndarray,
        vel: list[float] | np.ndarray,
        radius: np.float64,
        orientation: np.ndarray | None = None,
        angular_vel: np.ndarray | None = None,
    ):
        """
        Initialize a new physical body with translational state and physical properties.

        Parameters:
        -----------
        mass : np.float64
            The mass of the body. Must be strictly positive and non-zero to avoid
            singularities in force and acceleration calculations.
        pos : list[float] | np.ndarray
            Initial position of the body in 3D Cartesian space. Must be convertible
            to a NumPy array of shape (3,).
        vel : list[float] | np.ndarray
            Initial velocity vector of the body. Must be convertible to a NumPy
            array of shape (3,).
        radius : np.float64
            The physical collision radius of the body. Used for spatial culling
            and collision detection algorithms. Must be strictly positive.
        orientation : np.ndarray | None, optional
            Placeholder for the initial rotational orientation of the body,
            represented as a normalized quaternion of shape (4,). Currently
            unimplemented but reserved for future rigid-body dynamics (default is None).
        angular_vel : np.ndarray | None, optional
            Placeholder for the initial angular velocity vector of shape (3,).
            Currently unimplemented but reserved for future rigid-body dynamics
            (default is None).

        Raises:
        -------
        ValueError
            If mass or radius are non-positive or zero.
            If pos or vel cannot be resolved to exactly 3-dimensional vectors.
        """

        # ------------------------- Scalars -------------------------
        if mass <= 0:
            raise ValueError("Body mass must be positive and non-zero")
        self.mass = float(mass)

        if radius <= 0:
            raise ValueError("Body radius must be positive")
        self.radius = float(radius)

        # ------------------------- Translational State -------------------------
        self.pos = np.asarray(pos, dtype=np.float64)
        self.vel = np.asarray(vel, dtype=np.float64)

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
