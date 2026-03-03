"""
integrators.py

This module implements various numerical integration methods for solving ordinary differential equations (ODEs) that arise in physics simulations. It includes both non-symplectic integrators (like Euler and RK4) and symplectic integrators (like Velocity Verlet). The choice of integrator can affect the accuracy and stability of the simulation, especially for long-term integrations or systems with strong forces.

Functions:
- euler: Perform a single Euler step.
- rk4: Perform a single Runge-Kutta 4th order step.
- velocity_verlet: Perform a single Velocity Verlet step.

Helper functions:
- _rk4_step: Helper function that implements the RK4 algorithm.
- _derivative: Helper function that creates a derivative function for the ODE integrators based on the provided force function.
"""

import numpy as np


# ----------------------------- ODE integrators -----------------------------
def euler(masses, state_vector, force_fn, dt, t):
    """
    Perform a single Euler step.

    Parameters:
    -----------
    masses : list of float
        List of masses for the bodies in the system
    state_vector : np.ndarray
        Flattened array containing the positions and velocities of all bodies
    force_fn : function
        Function that computes the accelerations for the bodies based on their positions
    dt : float
        Time step for the integration
    t : float
        Current time

    Returns:
    --------
    np.ndarray
        Updated state vector after one Euler step

    Note: The Euler method is a simple first-order integration method and may not be suitable for long-term simulations or systems with strong forces due to its limited accuracy and stability.
    """
    n = len(masses)
    f = _derivative(masses, force_fn, n)
    return state_vector + f(t, state_vector) * dt


def rk4(masses, state_vector, force_fn, dt, t):
    """
    Perform a single Runge-Kutta 4th order step.

    Parameters:
    -----------
    masses : list of float
        List of masses for the bodies in the system
    state_vector : np.ndarray
        Flattened array containing the positions and velocities of all bodies
    force_fn : function
        Function that computes the accelerations for the bodies based on their positions
    dt : float
        Time step for the integration
    t : float
        Current time

    Returns:
    --------
    np.ndarray
        Updated state vector after one RK4 step

    Note: The RK4 method is a widely used fourth-order integration method that provides a good balance between accuracy and computational cost for many problems.
    """
    n = len(masses)

    f = _derivative(masses, force_fn, n)
    return _rk4_step(f, dt, t, state_vector)


def _rk4_step(fun, dt, t, x):
    """
    Perform a single RK4 step.

    Parameters:
    -----------
    fun : function
        Function that computes the derivative (dx/dt) given time t and state x
    dt : float
        Time step for the integration
    t : float
        Current time
    x : np.ndarray
        Current state vector (positions and velocities)

    Returns:
    --------
    np.ndarray
        Updated state vector after one RK4 step

    Note: This is a helper function that implements the RK4 algorithm. It computes intermediate slopes (k1, k2, k3, k4) and combines them to produce the final updated state vector.
    """
    k1 = fun(t, x)
    k2 = fun(t + dt / 2, x + dt / 2 * k1)
    k3 = fun(t + dt / 2, x + dt / 2 * k2)
    k4 = fun(t + dt, x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# ----------------------------- Symplectic integrators -----------------------------
def velocity_verlet(masses, state_vector, force_fn, dt, t):
    """
    Perform a single Velocity Verlet step.

    Parameters:
    -----------
    masses : list of float
        List of masses for the bodies in the system
    state_vector : np.ndarray
        Flattened array containing the positions and velocities of all bodies
    force_fn : function
        Function that computes the accelerations for the bodies based on their positions
    dt : float
        Time step for the integration
    t : float
        Current time

    Returns:
    --------
    np.ndarray
        Updated state vector after one Velocity Verlet step

    Note: The Velocity Verlet method is a symplectic integrator that is particularly well-suited for Hamiltonian systems, as it better conserves energy over long simulations compared to non-symplectic methods like Euler or RK4.
    It updates positions based on current velocities and accelerations, then computes new accelerations based on the updated positions, and finally updates velocities using the average of the old and new accelerations.
    """
    n = len(masses)
    pos = state_vector[: n * 3].reshape(n, 3)
    vel = state_vector[n * 3 :].reshape(n, 3)
    a_n = force_fn(masses, pos)
    pos_new = pos + vel * dt + 0.5 * a_n * dt**2
    a_new = force_fn(masses, pos_new)
    vel_new = vel + 0.5 * (a_n + a_new) * dt
    new_state = np.empty_like(state_vector)
    new_state[: n * 3] = pos_new.flatten()
    new_state[n * 3 :] = vel_new.flatten()
    return new_state


# ----------------------------- Helper functions -----------------------------
def _derivative(masses, force_fn, n):
    """
    Create a function that computes the derivative of the state vector (positions and velocities) given the current time and state.

    Parameters:
    -----------
    masses : list of float
        List of masses for the bodies in the system
    force_fn : function
        Function that computes the accelerations for the bodies based on their positions
    n : int
        Number of bodies in the system

    Returns:
    --------
    function
        A function that takes time t and state_vector as input and returns the derivative of the state vector (velocities and accelerations)

    Note: This function is used to create the derivative function needed for ODE integrators like Euler and RK4. It extracts positions and velocities from the state vector, computes accelerations using the provided force function, and returns the combined derivative vector.
    """

    def f(t, state_vector):
        """
        Compute the derivative of the state vector.

        Parameters:
        -----------
        t : float
            Current time
        state_vector : np.ndarray
            Flattened array containing the positions and velocities of all bodies

        Returns:
        --------
        np.ndarray
            Derivative of the state vector (velocities and accelerations)

        Note: The first half of the returned vector contains the velocities (derivative of positions), and the second half contains the accelerations (derivative of velocities) computed from the force function.
        """
        pos = state_vector[: n * 3].reshape(n, 3)
        vel = state_vector[n * 3 :].reshape(n, 3)
        accs = force_fn(masses, pos)
        return np.concatenate([vel.flatten(), accs.flatten()])

    return f
