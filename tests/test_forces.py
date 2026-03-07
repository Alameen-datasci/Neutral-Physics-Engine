"""
Unit tests for the force computation logic in the neutral_physics_engine package.

This module evaluates the `forces` function, which calculates the N-body gravitational
accelerations. The test suite ensures physical accuracy and numerical stability through:
* **Isolated Bodies:** Verifying that a single body experiences zero net acceleration.
* **Universal Gravitation:** Confirming two-body interactions adhere to Newton's law of universal gravitation.
* **Newton's Third Law:** Ensuring that in multi-body systems, forces are equal and opposite (net system force is zero).
* **Singularity Avoidance:** Checking that division-by-zero errors are prevented when the distance between bodies falls below the `EPS` threshold.
"""

import numpy as np
import pytest
from neutral_physics_engine.forces import forces, G, EPS


def test_single_body():
    """
    Test the forces function with a single body.
    This test verifies that when there is only one body in the simulation, the forces function returns zero acceleration for that body, as there are no other bodies to exert a gravitational force on it.
    The test checks that the output is an array of zeros with the correct shape, confirming that the function handles the case of a single body correctly and does not produce any unintended forces or errors in this scenario.
    This is an important edge case to test, as it ensures that the forces function can handle the simplest possible scenario without issues, which is a fundamental requirement for the correctness and robustness of the physics engine.
    """
    masses = [1.989e30]  # Mass of the Sun
    pos = np.array([[0.0, 0.0, 0.0]])
    accs = forces(masses, pos)
    # Assert the output shape is correct and all values are strictly zero
    assert accs.shape == (1, 3)
    np.testing.assert_allclose(accs, np.zeros((1, 3)))


def test_two_bodies():
    """
    Test the forces function with two bodies.
    This test verifies that the forces function correctly computes the gravitational accelerations for two bodies based on their masses and positions.
    The test sets up a simple scenario with two bodies of equal mass placed at a known distance apart, and checks that the computed accelerations match the expected values derived from Newton's law of universal gravitation.
    The expected accelerations are calculated using the formula a = G * m / r^2, where G is the gravitational constant, m is the mass of the other body, and r is the distance between the two bodies.
    The test ensures that the forces function correctly implements this formula and that the accelerations are equal in magnitude and opposite in direction, as required by Newton's third law of motion.
    This test is crucial for confirming that the core functionality of the forces function is working correctly for a basic two-body scenario, which is a fundamental building block for more complex simulations involving multiple bodies.
    """
    masses = [1e10, 1e10]
    # Place body 0 at origin, body 1 at x = 1.0 meters
    pos = np.array([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    # Compute the accelerations using the forces function
    accs = forces(masses, pos)
    expected_acc = G * 1e10 / (1.0**2)
    # The expected accelerations should be equal in magnitude and opposite in direction
    expected_accs = np.array([[expected_acc, 0.0, 0.0], [-expected_acc, 0.0, 0.0]])

    np.testing.assert_allclose(accs, expected_accs, rtol=1e-7)


def test_newtons_third_law():
    """
    Test that the forces function satisfies Newton's third law of motion.
    This test verifies that the forces function correctly implements Newton's third law, which states that for every action, there is an equal and opposite reaction. In the context of gravitational forces, this means that the force (and therefore the acceleration) exerted by body A on body B should be equal in magnitude and opposite in direction to the force exerted by body B on body A.
    The test sets up a scenario with three bodies of varying masses and positions, and checks that the net force on the system is zero, which is a direct consequence of Newton's third law. This ensures that the forces function correctly calculates the interactions between multiple bodies and that the overall momentum of the system is conserved.
    This test is important for confirming that the forces function not only computes the correct magnitudes of the accelerations but also correctly accounts for the directions of the forces, which is essential for the physical accuracy of the simulations produced by the physics engine.
    """
    # Create a system of three bodies with varying masses and positions
    masses = [1e10, 3e10, 5e10]
    # Place the bodies at different positions in space
    pos = np.array([[0.0, 0.0, 0.0], [10.0, 5.0, -2.0], [-4.0, 8.0, 12.0]])
    # Compute the accelerations using the forces function
    accs = forces(masses, pos)
    # Calculate forces: F_i = m_i * a_i
    forces_vectors = np.array(masses)[:, np.newaxis] * accs
    # Net force on the entire system should be zero
    net_force = np.sum(forces_vectors, axis=0)

    np.testing.assert_allclose(net_force, np.array([0.0, 0.0, 0.0]), atol=1e-7)


def test_singularity_avoidance():
    """
    Test that the forces function avoids singularities when bodies are placed too close together.
    This test ensures that the forces function correctly handles cases where two bodies are positioned at a distance less than the threshold EPS, preventing division by zero or other numerical instabilities.
    """
    masses = [1.0, 1.0]
    # Place bodies closer than EPS (1e-10)
    pos = np.array([[0.0, 0.0, 0.0], [EPS * 0.5, 0.0, 0.0]])
    # Compute the accelerations using the forces function
    accs = forces(masses, pos)
    # Accelerations should be completely skipped and remain zero
    np.testing.assert_allclose(accs, np.zeros((2, 3)))
