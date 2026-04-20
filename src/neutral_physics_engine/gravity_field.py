"""
gravity_field.py

This module provides the GravityField class, which acts as a high-level interface
for computing gravitational accelerations and potential energies in N-body simulations.
It leverages the Barnes-Hut octree implementation to efficiently evaluate forces
across large ensembles of bodies.

By encapsulating the tree-building process and caching the tree structure per
position state, GravityField ensures optimal performance during simulation
time-stepping loops. It avoids redundant tree constructions when calculating
multiple properties (e.g., forces and potentials) for the exact same spatial configuration.
"""

from neutral_physics_engine.octree import Octree
import numpy as np


class GravityField:
    """
    Evaluator for gravitational fields using the Barnes-Hut approximation.

    The GravityField class manages the calculation of gravitational accelerations
    and potential energies for a system of massive bodies. It automatically handles
    the construction and internal caching of the underlying Octree structure to
    minimize computational overhead during a single simulation time step.
    """

    def __init__(self, theta: np.float64 = 0.5):
        """
        Initialize the GravityField evaluator.

        Parameters:
        -----------
        theta : np.float64, optional
            The Barnes-Hut opening angle parameter (default 0.5). This controls
            the accuracy-performance trade-off. Lower values yield higher accuracy
            by traversing deeper into the tree, while higher values improve execution speed.
        """
        self.theta = theta
        self._cached_pos = None
        self._cached_tree = None

    def _get_tree(
        self, masses: list[float] | np.ndarray, positions: np.ndarray
    ) -> Octree:
        """
        Retrieve or build the Barnes-Hut octree for the current system state.

        This internal method caches the tree based on the provided positions. If the
        positions strictly match the cached state, the existing tree is returned,
        preventing redundant and expensive rebuilds.

        Parameters:
        -----------
        masses : list[float] | np.ndarray
            Array-like structure containing the masses of the N bodies.
        positions : np.ndarray
            A 2D array of shape (N, 3) containing the 3D position vectors of the bodies.

        Returns:
        --------
        Octree
            The constructed or cached octree representing the spatial distribution.
        """
        if self._cached_tree is not None and np.array_equal(
            self._cached_pos, positions
        ):
            return self._cached_tree

        self._cached_pos = positions.copy()
        self._cached_tree = Octree(masses, positions, theta=self.theta)
        self._cached_tree.build()
        return self._cached_tree

    def __call__(
        self, masses: list[float] | np.ndarray, positions: np.ndarray
    ) -> np.ndarray:
        """
        Compute the gravitational acceleration vector for every body in the system.

        This allows the GravityField instance to be called directly as a function,
        yielding the total approximate acceleration on each particle due to all
        other particles in the system.

        Parameters:
        -----------
        masses : list[float] | np.ndarray
            Array-like structure containing the masses of the N bodies.
        positions : np.ndarray
            A 2D array of shape (N, 3) containing the 3D position vectors.

        Returns:
        --------
        np.ndarray
            A 2D array of shape (N, 3) containing the net acceleration vectors
            for each corresponding body.
        """
        tree = self._get_tree(masses, positions)
        accs = np.zeros_like(positions, dtype=np.float64)
        for i in range(len(masses)):
            accs[i] = tree.compute_acceleration(i)
        return accs

    def compute_potential(
        self, masses: list[float] | np.ndarray, positions: np.ndarray
    ) -> np.float64:
        """
        Compute the total gravitational potential energy of the N-body system.

        Calculates the sum of the potential energies of all particles using the
        Barnes-Hut approximation. The method ensures the tree is built and
        evaluates the per-body potential, multiplying by half the mass to avoid
        double-counting particle pairs.

        Parameters:
        -----------
        masses : list[float] | np.ndarray
            Array-like structure containing the masses of the N bodies.
        positions : np.ndarray
            A 2D array of shape (N, 3) containing the 3D position vectors.

        Returns:
        --------
        np.float64
            The total gravitational potential energy of the entire system.
        """
        tree = self._get_tree(masses, positions)
        PE = 0.0
        for i in range(len(masses)):
            PE += 0.5 * masses[i] * tree.compute_potential(i)
        return np.float64(PE)
