"""
integrators.py

This module implements discrete-time numerical integration schemes for solving the
ordinary differential equations (ODEs) governing N-body dynamics. It provides both
standard non-symplectic Runge-Kutta class integrators (Euler, RK4) and a symplectic
integrator (Velocity Verlet).

The choice of integrator dictates the stability, computational cost, and long-term
energy conservation characteristics of the simulation:
- Non-symplectic integrators (Euler, RK4) can suffer from energy drift over time
  but offer varying degrees of local truncation accuracy.
- Symplectic integrators (Velocity Verlet) explicitly preserve the geometric
  properties of Hamiltonian systems, ensuring bounded energy fluctuations over
  arbitrarily long integration periods, making them ideal for orbital mechanics.
"""

import numpy as np
from typing import Callable


# ----------------------------- ODE integrators -----------------------------
def euler(
    masses: list[float],
    state_vector: list | np.ndarray,
    field: Callable[[list[float], np.ndarray], np.ndarray],
    dt: np.float64,
    t: np.float64,
) -> np.ndarray:
    """
    Propagate the system state forward in time using the first-order Forward Euler method.

    This is an explicit, non-symplectic integrator with a local truncation error
    proportional to O(dt^2). While computationally inexpensive, it is generally
    unstable for long-term integration of conservative forces and will exhibit
    rapid energy divergence.

    Parameters:
    -----------
    masses : list[float]
        List of masses for the N bodies in the system.
    state_vector : np.ndarray
        A flattened 1D array of length 6N containing the concatenated positions
        and velocities of all bodies.
    field : Callable
        A callable vector field (e.g., GravityField) that accepts masses and
        positions, returning the instantaneous acceleration array.
    dt : np.float64
        The temporal step size for the integration.
    t : np.float64
        The current simulation time.

    Returns:
    --------
    np.ndarray
        The updated 1D state vector after one Euler time step.
    """
    n = len(masses)
    f = _derivative(masses, field, n)
    return state_vector + f(t, state_vector) * dt


def rk4(
    masses: list[float],
    state_vector: list | np.ndarray,
    field: Callable[[list[float], np.ndarray], np.ndarray],
    dt: np.float64,
    t: np.float64,
) -> np.ndarray:
    """
    Propagate the system state forward using the classical 4th-order Runge-Kutta method (RK4).

    RK4 evaluates the phase-space derivative at four distinct points within the
    time interval to compute a highly accurate weighted average slope. It has a local
    truncation error of O(dt^5). Though non-symplectic (energy will eventually drift),
    its high precision makes it robust for transient dynamics or short-timescale accuracy.

    Parameters:
    -----------
    masses : list[float]
        List of masses for the N bodies in the system.
    state_vector : np.ndarray
        A flattened 1D array of length 6N containing the concatenated positions
        and velocities of all bodies.
    field : Callable
        A callable vector field that computes accelerations given the spatial configuration.
    dt : np.float64
        The temporal step size for the integration.
    t : np.float64
        The current simulation time.

    Returns:
    --------
    np.ndarray
        The updated 1D state vector after one RK4 time step.
    """
    n = len(masses)

    f = _derivative(masses, field, n)
    return _rk4_step(f, dt, t, state_vector)


def _rk4_step(fun, dt: np.float64, t: np.float64, x: np.ndarray) -> np.ndarray:
    """
    Internal kernel to evaluate a single RK4 integration step.

    Parameters:
    -----------
    fun : Callable
        The phase-space derivative function computing dx/dt.
    dt : np.float64
        The time step size.
    t : np.float64
        The current time.
    x : np.ndarray
        The current state vector.

    Returns:
    --------
    np.ndarray
        The updated state vector resulting from the RK4 weighted slopes (k1-k4).
    """
    k1 = fun(t, x)
    k2 = fun(t + dt / 2, x + dt / 2 * k1)
    k3 = fun(t + dt / 2, x + dt / 2 * k2)
    k4 = fun(t + dt, x + dt * k3)
    return x + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


# ----------------------------- Symplectic integrators -----------------------------
def velocity_verlet(
    masses: list[float],
    state_vector: list | np.ndarray,
    field: Callable[[list[float], np.ndarray], np.ndarray],
    dt: np.float64,
    t: np.float64,
) -> np.ndarray:
    """
    Propagate the system state using the Velocity Verlet symplectic integrator.

    Velocity Verlet is a second-order symplectic method specifically designed for
    conservative Hamiltonian systems (like gravity). By staggering the velocity
    and position updates, it precisely preserves the phase-space volume, resulting
    in exceptional long-term energy stability without the computational overhead
    of higher-order methods like RK4.

    Parameters:
    -----------
    masses : list[float]
        List of masses for the N bodies in the system.
    state_vector : np.ndarray
        A flattened 1D array of length 6N containing the concatenated positions
        and velocities of all bodies.
    field : Callable
        A callable vector field that computes accelerations given the spatial configuration.
    dt : np.float64
        The temporal step size for the integration.
    t : np.float64
        The current simulation time.

    Returns:
    --------
    np.ndarray
        The updated 1D state vector after one Velocity Verlet time step.
    """
    n = len(masses)
    pos = state_vector[: n * 3].reshape(n, 3)
    vel = state_vector[n * 3 :].reshape(n, 3)
    a_n = field(masses, pos)
    pos_new = pos + vel * dt + 0.5 * a_n * dt**2
    a_new = field(masses, pos_new)
    vel_new = vel + 0.5 * (a_n + a_new) * dt
    new_state = np.empty_like(state_vector)
    new_state[: n * 3] = pos_new.flatten()
    new_state[n * 3 :] = vel_new.flatten()
    return new_state


# ----------------------------- Helper functions -----------------------------
def _derivative(
    masses: list[float], field: Callable[[list[float], np.ndarray], np.ndarray], n: int
):
    """
    Generate a phase-space derivative map for non-symplectic integrators.

    This function closes over the system properties (masses, force field, and body count)
    to produce a standard first-order ODE derivative function compliant with generic
    integration schemes like RK4.

    Parameters:
    -----------
    masses : list[float]
        List of body masses.
    field : Callable
        Function calculating instantaneous acceleration.
    n : int
        The total number of bodies.

    Returns:
    --------
    Callable
        A closure function `f(t, state_vector)` that computes the coupled time
        derivatives of positions and velocities.
    """

    def f(t: np.float64, state_vector: np.ndarray) -> np.ndarray:
        """
        Evaluate the phase-space time derivative mapping.

        Parameters:
        -----------
        t : np.float64
            Current simulation time.
        state_vector : np.ndarray
            Flattened state vector (positions and velocities).

        Returns:
        --------
        np.ndarray
            Flattened derivative vector where the first half contains velocities
            (d/dt positions) and the second half contains accelerations (d/dt velocities).
        """
        pos = state_vector[: n * 3].reshape(n, 3)
        vel = state_vector[n * 3 :].reshape(n, 3)
        accs = field(masses, pos)
        return np.concatenate([vel.flatten(), accs.flatten()])

    return f
