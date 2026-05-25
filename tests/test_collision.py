"""
Unit tests for the CollisionSystem.
"""

import numpy as np
import pytest
from neutral_physics_engine.collision import CollisionSystem


def test_elastic_head_on_collision():
    """
    Test a perfectly elastic (restitution=1.0) head-on collision between
    two bodies of equal mass. They should perfectly swap velocities.
    """
    masses = np.array([1.0, 1.0])
    positions = np.array([[-49.0, 0.0, 0.0], [49.0, 0.0, 0.0]])
    velocities = np.array([[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    radii = np.array([0.5, 0.5])

    system = CollisionSystem(masses, positions, velocities, radii, restitution=1.0)

    system.resolve_collisions()
    # They should exchange velocities in an elastic collision, so the first body should now be moving to the right and the second body should be moving to the left
    np.testing.assert_allclose(velocities[0], [-1.0, 0.0, 0.0])
    np.testing.assert_allclose(velocities[1], [1.0, 0.0, 0.0])

    dist = np.linalg.norm(positions[0] - positions[1])
    assert dist >= 1.0


def test_inelastic_collision():
    """
    Test a completely inelastic collision (restitution=0.0).
    The bodies should stick together and have a final relative velocity of zero.
    """
    masses = np.array([1.0, 1.0])
    positions = np.array([[-0.4, 0.0, 0.0], [0.4, 0.0, 0.0]])
    velocities = np.array([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]])
    radii = np.array([0.5, 0.5])

    system = CollisionSystem(masses, positions, velocities, radii, restitution=0.0)

    system.resolve_collisions()
    # In a perfectly inelastic collision, the two bodies should stick together and come to rest, so both velocities should be zero
    # Initial momentum = 0 and Final momentum = 2v, so v = 0
    np.testing.assert_allclose(velocities[0], [0.0, 0.0, 0.0], rtol=1e-10)
    np.testing.assert_allclose(velocities[1], [0.0, 0.0, 0.0], rtol=1e-10)
