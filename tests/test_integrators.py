"""
Unit tests for the numerical integrators in the neutral_physics_engine package.

This module verifies the correctness of the integration algorithms used to calculate
the kinematic state (positions and velocities) of bodies over discrete time steps.
It uses a controlled, constant-acceleration dummy force to validate:
* **State Derivatives:** The internal `_derivative` calculation mapping velocities and accelerations.
* **Euler Integration:** First-order integration accuracy.
* **Runge-Kutta (RK4) Integration:** Fourth-order integration accuracy.
* **Velocity Verlet Integration:** Symplectic integration accuracy, typically used for energy conservation.
"""

from neutral_physics_engine.integrators import euler, rk4, velocity_verlet, _derivative
import numpy as np
import pytest


def dummy_force_fn(masses, positions):
    """
    A dummy force function that returns a constant acceleration for testing purposes.
    This function is used to test the integrators with a simple scenario where the acceleration is constant and does not depend on the positions or masses of the bodies.
    It allows us to verify that the integrators correctly update the positions and velocities based on the given accelerations, without the complexity of a more realistic force model.
    The constant acceleration in the x-direction simplifies the expected results, making it easier to calculate the expected new state after one integration step and to verify the correctness of the integrators' implementations.
    """
    n = len(masses)
    accs = np.zeros((n, 3))
    accs[:, 0] = 2.0  # Constant acceleration in x-direction
    return accs


def test_derivative():
    """
    Test the derivative function for the integrators.
    This test verifies that the derivative function correctly computes the derivatives of the state vector based on the given masses, positions, and the dummy force function.
    The state vector is structured to include the positions and velocities of two bodies, and the dummy force function provides a constant acceleration in the x-direction for both bodies.
    The expected output of the derivative function is a new state vector that includes the velocities (which are the derivatives of the positions) and the accelerations (which are the derivatives of the velocities) for both bodies.
    This test ensures that the derivative function is correctly implemented and that it can be used by the integrators to update the state of the simulation accurately based on the forces acting on the bodies.
    """
    masses = [1.0, 2.0]
    # State vector: [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2]
    state = np.array([0.0, 0.0, 0.0, 10, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Compute the derivative using the dummy force function
    f = _derivative(masses, dummy_force_fn, len(masses))
    deriv = f(0, state)
    # Expected derivative: [vx1, vy1, vz1, vx2, vy2, vz2, ax1, ay1, az1, ax2, ay2, az2]
    expected = np.array([10, 0.0, 0.0, 0.0, 0.0, 0.0, 2.0, 0.0, 0.0, 2.0, 0.0, 0.0])
    np.testing.assert_array_equal(deriv, expected)


def test_euler_constant_acceleration():
    """
    Test the Euler integrator with a constant acceleration.
    This test verifies that the Euler integrator correctly updates the positions and velocities of the bodies based on a constant acceleration provided by the dummy force function.
    The initial state includes the positions and velocities of two bodies, and the dummy force function provides a constant acceleration in the x-direction for both bodies.
    The expected new state after one Euler step is calculated based on the initial velocities and the constant acceleration, and the test checks that the output from the Euler integrator matches this expected state.
    This ensures that the Euler integrator is correctly implemented and that it can be used to update the state of the simulation accurately based on the forces acting on the bodies, even in a simple scenario with constant acceleration.
    """
    masses = [1.0, 2.0]
    # State vector: [x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, vx2, vy2, vz2]
    state = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Time step and current time for the integration
    dt = 0.1
    t = 0.0
    # Compute the new state using the Euler integrator with the dummy force function
    new_state = euler(masses, state, dummy_force_fn, dt, t)
    # Expected new state after one Euler step
    expected = np.array(
        [1.01, 0.0, 0.0, 10.01, 0.0, 0.0, 10.2, 0.0, 0.0, 0.2, 0.0, 0.0]
    )
    np.testing.assert_allclose(new_state, expected, rtol=1e-17, atol=1e-2)


def test_rk4_constant_acceleration():
    """
    Test the RK4 integrator with a constant acceleration.
    This test verifies that the RK4 integrator correctly updates the positions and velocities of the bodies based on a constant acceleration provided by the dummy force function.
    The initial state includes the positions and velocities of two bodies, and the dummy force function provides a constant acceleration in the x-direction for both bodies.
    The expected new state after one RK4 step is calculated based on the initial velocities and the constant acceleration, and the test checks that the output from the RK4 integrator matches this expected state.
    This ensures that the RK4 integrator is correctly implemented and that it can be used to update the state of the simulation accurately based on the forces acting on the bodies, even in a simple scenario with constant acceleration.
    """
    masses = [1.0, 2.0]
    # State vector: [x1, y1, z1, vx1, vy1, vz1, x2, y2, z2, vx2, vy2, vz2]
    state = np.array([0.0, 0.0, 0.0, 10, 0.0, 0.0, 10.0, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Time step and current time for the integration
    dt = 0.1
    t = 0.0
    # Compute the new state using the RK4 integrator with the dummy force function
    new_state = rk4(masses, state, dummy_force_fn, dt, t)
    # Expected new state after one RK4 step (same as Euler for constant acceleration)
    expected = np.array(
        [1.01, 0.0, 0.0, 10.01, 0.0, 0.0, 10.2, 0.0, 0.0, 0.2, 0.0, 0.0]
    )
    np.testing.assert_allclose(new_state, expected, rtol=1e-15, atol=1e-15)


def test_velocity_verlet_constant_acceleration():
    """
    Test the Velocity Verlet integrator with a constant acceleration.
    This test verifies that the Velocity Verlet integrator correctly updates the positions and velocities of the bodies based on a constant acceleration provided by the dummy force function.
    The initial state includes the positions and velocities of two bodies, and the dummy force function provides a constant acceleration in the x-direction for both bodies.
    The expected new state after one Velocity Verlet step is calculated based on the initial velocities and the constant acceleration, and the test checks that the output from the Velocity Verlet integrator matches this expected state.
    This ensures that the Velocity Verlet integrator is correctly implemented and that it can be used to update the state of the simulation accurately based on the forces acting on the bodies, even in a simple scenario with constant acceleration.
    """
    masses = [1.0, 2.0]
    # State vector: [x1, y1, z1, x2, y2, z2, vx1, vy1, vz1, vx2, vy2, vz2]
    state = np.array([0.0, 0.0, 0.0, 10.0, 0.0, 0.0, 10, 0.0, 0.0, 0.0, 0.0, 0.0])
    # Time step and current time for the integration
    dt = 0.1
    t = 0.0
    # Compute the new state using the Velocity Verlet integrator with the dummy force function
    new_state = velocity_verlet(masses, state, dummy_force_fn, dt, t)
    # Expected new state after one Velocity Verlet step
    expected = np.array(
        [1.01, 0.0, 0.0, 10.01, 0.0, 0.0, 10.2, 0.0, 0.0, 0.2, 0.0, 0.0]
    )
    np.testing.assert_allclose(new_state, expected)
