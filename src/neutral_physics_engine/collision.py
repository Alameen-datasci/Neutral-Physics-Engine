"""
collision.py

This module implements the collision detection and resolution system for the
neutral_physics_engine. It is designed to handle perfectly elastic to perfectly
inelastic collisions between spherical bodies in a 3D environment.

The system employs a two-phase approach:
- Broad-phase & Narrow-phase Detection: Leverages the O(N log N) Barnes-Hut
   Octree to efficiently cull distant bodies and identify intersecting bounding
   spheres, bypassing the O(N^2) naive pairwise check.
- Resolution: Utilizes an impulse-based constraint solver to instantaneously
   update velocities based on the coefficient of restitution. It also applies a
   post-resolution positional correction step to mitigate floating-point
   inaccuracies and prevent bodies from permanently interpenetrating or "sinking"
   into one another.

Dependencies:
    numpy : Matrix and vector mathematics
    neutral_physics_engine.octree : High-performance spatial partitioning
"""

from neutral_physics_engine.octree import Octree
import numpy as np


class CollisionSystem:
    """
    Manages the detection and physical resolution of rigid-body collisions.

    This class computes the required impulses to separate colliding bodies and
    updates their kinematic state (velocities and positions) in place. It relies
    on the Octree spatial structure to optimize pair finding for large N.
    """

    def __init__(
        self,
        masses: np.ndarray,
        positions: np.ndarray,
        velocities: np.ndarray,
        radii: np.ndarray,
        restitution: np.float64 = 0.8,
    ):
        """
        Initialize the collision system with the current kinematic state.

        Parameters:
        -----------
        masses : np.ndarray
            1D array of body masses (shape: (N,)).
        positions : np.ndarray
            2D array of current body positions (shape: (N, 3)).
        velocities : np.ndarray
            2D array of current body velocities (shape: (N, 3)).
        radii : np.ndarray
            1D array of body bounding-sphere radii used for collision
            geometry (shape: (N,)).
        restitution : float, optional
            Coefficient of restitution (e) defining the elasticity of the
            collisions. 1.0 represents a perfectly elastic collision (no kinetic
            energy lost), while 0.0 represents a perfectly inelastic collision
            (bodies stick together). Default is 0.8.

        Raises:
        -------
        ValueError
            If input arrays are None, mismatched in length (N), incorrectly
            shaped for 3D space, or if the restitution is out of the [0, 1] bounds.
        """
        if masses is None or positions is None or velocities is None or radii is None:
            raise ValueError("...")

        N = len(masses)
        if positions.shape[0] != N or velocities.shape[0] != N or len(radii) != N:
            raise ValueError("...")

        if positions.shape[1] != 3 or velocities.shape[1] != 3:
            raise ValueError("...")

        if restitution < 0 or restitution > 1:
            raise ValueError("...")

        self.masses = masses
        self.positions = positions
        self.velocities = velocities
        self.radii = radii
        self.restitution = restitution

    def _get_collision_pairs(self) -> list[tuple[int, int]]:
        """
        Identify all unique pairs of intersecting bodies.

        Builds a spatial Octree from the current simulation state and traverses
        it to find bodies whose combined radii are greater than their distance.

        Returns:
        --------
        list[tuple[int, int]]
            A list of index pairs (i, j) representing colliding bodies.
        """
        # creating octree
        tree = Octree(self.masses, self.positions, self.radii)
        tree.build()
        return tree.find_collisions()

    def resolve_collisions(self) -> None:
        """
        Process all identified collision pairs and apply physics resolutions.

        For each colliding pair, this method:
        1. Computes the collision normal.
        2. Calculates the relative velocity to ensure bodies are approaching.
        3. Applies an instantaneous impulse to alter velocities based on mass
           and the coefficient of restitution.
        4. Applies a slight positional correction to separate overlapping
           spheres, ensuring system stability over long integrations.

        Note:
        -----
        Modifies `self.velocities` and `self.positions` in-place.
        """
        pairs = self._get_collision_pairs()

        for i, j in pairs:
            p_i, p_j = self.positions[i], self.positions[j]
            v_i, v_j = self.velocities[i], self.velocities[j]
            m_i, m_j = self.masses[i], self.masses[j]
            r_i, r_j = self.radii[i], self.radii[j]

            delta = np.subtract(p_j, p_i)
            dist = np.linalg.norm(delta)
            if dist < 1e-12:
                n_hat = np.array([1.0, 0.0, 0.0])
                dist = 1e-12
            else:
                n_hat = delta / dist

            v_rel = np.subtract(v_j, v_i)
            # check if bodies are moving apart
            if np.dot(v_rel, n_hat) > 0:
                continue

            inv_mass_sum = (1 / m_j) + (1 / m_i)
            J = -(1 + self.restitution) * np.dot(v_rel, n_hat) / inv_mass_sum

            # update velocities
            self.velocities[i] -= (J / m_i) * n_hat
            self.velocities[j] += (J / m_j) * n_hat

            # positional correnction to prevent interpenetration
            penetration = r_i + r_j - dist
            slop = 1e-5
            if penetration > slop:
                percent = 0.8
                correction = max(penetration - slop, 0) / inv_mass_sum * percent
                # updating positions
                self.positions[i] -= (correction / m_i) * n_hat
                self.positions[j] += (correction / m_j) * n_hat
