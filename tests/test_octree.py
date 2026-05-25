"""
Unit tests for the Barnes-Hut Octree implementation.
"""
import numpy as np
import pytest
from neutral_physics_engine.octree import Octree


def test_octree_build_and_com():
    """
    Test that the octree correctly accumulates total mass and calculates
    the center of mass for a symmetric distribution.
    """
    # Create a simple configuration of masses and positions to test the octree's build and center of mass calculation
    masses = np.array([1.0, 1.0])
    positions = np.array([[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]])

    tree = Octree(masses, positions)
    tree.build()
    # The total mass should be the sum of the individual masses
    assert tree.root.mass == 2.0
    # The center of mass for two equal masses at symmetric positions should be at the origin
    np.testing.assert_array_equal(tree.root.com, np.array([0.0, 0.0, 0.0]))


def test_octree_point_mass_acceleration():
    """
    Test that the octree calculates the acceleration on a test point correctly
    compared to direct summation.
    """
    # Create a simple configuration of masses and positions to test the octree's acceleration calculation
    masses = np.array([1.0, 10.0, 10.0])
    positions = np.array([[100.0, 0.0, 0.0], [0.1, 0.0, 0.0], [-0.1, 0.0, 0.0]])

    tree = Octree(masses, positions, theta=0.5)
    tree.build()
    # Compare the acceleration calculated by the octree with direct summation
    tree_acc, direct_acc, diff = tree.compare_with_direct(0)
    # The acceleration from the octree should be close to the direct calculation, within a reasonable tolerance
    np.testing.assert_allclose(tree_acc, direct_acc, rtol=1e-3)
