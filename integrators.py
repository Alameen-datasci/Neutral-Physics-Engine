"""
Numerical integration methods.

Implements Euler and Runge-Kutta (RK4) time-stepping
for advancing the system state.
"""

import numpy as np

def euler_step(bodies, force_fn, dt, t):
    """
    Update the body using Euler integration.

    Parameters:
        body : Body
            Object to update
        force_fn : function
            Returns acceleration vector
        dt : float
            Time step
    """
    accs = force_fn(bodies)
    for i, b in enumerate(bodies):
        b.vel += accs[i] * dt
        b.pos += b.vel * dt

def rk4_step(fun, dt, t, x):
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

    The RK4 method computes intermediate slopes (k1, k2, k3, k4) to achieve
    higher accuracy compared to simpler methods like Euler's method.
    """
    k1 = fun(t, x)
    k2 = fun(t + dt/2, x + dt/2 * k1)
    k3 = fun(t + dt/2, x + dt/2 * k2)
    k4 = fun(t + dt, x + dt * k3)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def derivative(bodies, force_fn):
    """
    Create a function that computes the derivative of the state vector for RK4.

    Parameters:
    -----------
    bodies : list of Body
        List of bodies in the simulation
    force_fn : function
        Function that computes the accelerations for the bodies

    Returns:
    --------
    function
        A function that takes time t and state vector x, and returns the derivative dx/dt

    This function is used to convert the list of Body objects and the force function
    into a format suitable for the RK4 integrator, which operates on flat state vectors.
    """
    n = len(bodies)
    dim = bodies[0].pos.size
    
    def f(t, x):
        """
        Compute the derivative of the state vector.

        Parameters:
        -----------
        t : float
            Current time (not used in this case, but included for generality)
        x : np.ndarray
            Current state vector (positions and velocities) of shape (2*n*dim,)

        Returns:
        --------
        np.ndarray
            Derivative of the state vector (dx/dt) of shape (2*n*dim,)

        The state vector x is structured as follows:
        - The first n*dim elements correspond to the positions of the bodies (flattened).
        - The next n*dim elements correspond to the velocities of the bodies (flattened).
        """
        pos = x[:n*dim].reshape(n, dim)
        vel = x[n*dim:].reshape(n, dim)

        for i in range(n):
            bodies[i].pos = pos[i]
            bodies[i].vel = vel[i]

        accs = force_fn(bodies)

        dx = np.zeros_like(x)
        dx[:n*dim] = vel.flatten()
        dx[n*dim:] = np.array(accs).flatten()
        return dx

    return f

def rk4_body_step(bodies, force_fn, dt, t):
    """
    Update the bodies using RK4 integration.

    Parameters:
    -----------
    bodies : list of Body
        List of bodies to update
    force_fn : function
        Function that computes the accelerations for the bodies
    dt : float
        Time step for the integration
    t : float
        Current time

    This function prepares the state vector from the list of Body objects, computes the derivative function,
    and then performs a single RK4 step to update the positions and velocities of the bodies.

    After the RK4 step, it reshapes the updated state vector back into positions and velocities and updates the Body objects accordingly.
    """
    n = len(bodies)
    dim = bodies[0].pos.size

    x0 = np.concatenate(
        [np.array([b.pos for b in bodies]).flatten(),
         np.array([b.vel for b in bodies]).flatten()]
    )

    f = derivative(bodies, force_fn)
    x1 = rk4_step(f, dt, t, x0)

    pos = x1[:n*dim].reshape(n, dim)
    vel = x1[n*dim:].reshape(n, dim)

    for i in range(n):
        bodies[i].pos = pos[i]
        bodies[i].vel = vel[i]