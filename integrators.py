"""
Numerical integration methods.

Implements Euler and Runge-Kutta (RK4) time-stepping
for advancing the system state.
"""

import numpy as np

def euler_step(bodies, force_fn, dt, t):
    """
    Update the bodies using Euler integration.

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

    This function computes the accelerations for each body using the provided force function, and then updates the positions and velocities of the bodies using the Euler method. The positions are updated based on the current velocities, and the velocities are updated based on the computed accelerations.

    Note: The Euler method is a simple first-order integration method and may not be suitable for long-term simulations or systems with strong forces due to its limited accuracy and stability.
    """
    n = len(bodies)
    masses = np.array([b.mass for b in bodies])

    x0 = np.concatenate(
        [np.array([b.pos for b in bodies]).flatten(),
         np.array([b.vel for b in bodies]).flatten()]
    )

    f = derivative(masses, force_fn, n)
    x1 = x0 + f(t, x0) * dt

    pos = x1[:n*3].reshape(n, 3)
    vel = x1[n*3:].reshape(n, 3)

    for i in range(n):
        bodies[i].pos = pos[i]
        bodies[i].vel = vel[i]

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

def derivative(masses, force_fn, n):
    """
    Create a function that computes the derivative of the state vector.

    Parameters:
    -----------
    masses : np.ndarray
        Array of masses for each body in the simulation
    force_fn : function
        Function that computes the accelerations for the bodies
    n : int
        Number of bodies in the simulation

    Returns:
    --------
    function
        A function that takes time t and state vector x, and returns the derivative dx/dt
    
    This function constructs a derivative function that can be used with the RK4 integrator. The returned function computes the accelerations based on the current positions of the bodies and returns the time derivative of the state vector, which includes both the velocities (derivative of positions) and the accelerations (derivative of velocities).

    The state vector x is expected to be a concatenation of the positions and velocities of all bodies, and the derivative function will return a vector of the same shape containing the derivatives of these quantities.
    """
    
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
        pos = x[:n*3].reshape(n, 3)
        vel = x[n*3:].reshape(n, 3)
        accs = force_fn(masses, pos)
        dx = np.zeros_like(x)
        dx[:n*3] = vel.flatten()
        dx[n*3:] = accs.flatten()
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

    masses = np.array([b.mass for b in bodies])

    x0 = np.concatenate(
        [np.array([b.pos for b in bodies]).flatten(),
         np.array([b.vel for b in bodies]).flatten()]
    )

    f = derivative(masses, force_fn, n)
    x1 = rk4_step(f, dt, t, x0)

    pos = x1[:n*3].reshape(n, 3)
    vel = x1[n*3:].reshape(n, 3)

    for i in range(n):
        bodies[i].pos = pos[i]
        bodies[i].vel = vel[i]

def velocity_verlet_step(bodies, force_fn, dt, t):
    """
    Update the bodies using the Velocity Verlet integration method.

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
    
    The Velocity Verlet method is a symplectic integrator that provides better energy conservation properties compared to the Euler method, especially for systems with conservative forces. It updates positions and velocities in a way that takes into account the accelerations at both the current and the next time step, leading to improved stability and accuracy for many physical simulations.

    The method first computes the new positions using the current velocities and accelerations, then computes the new accelerations based on the updated positions, and finally updates the velocities using the average of the old and new accelerations.

    Note: The Velocity Verlet method is particularly well-suited for simulations of classical mechanics, such as planetary motion or molecular dynamics, where energy conservation is important.
    """
    n = len(bodies)
    masses = np.array([b.mass for b in bodies])
    pos = np.array([b.pos for b in bodies])
    vel = np.array([b.vel for b in bodies])
    a_n = force_fn(masses, pos)                         # Compute current accelerations based on current positions

    pos_new = pos + vel * dt + 0.5 * a_n * dt**2        # Update positions using current velocities and accelerations

    a_new = force_fn(masses, pos_new)                   # Compute new accelerations based on updated positions

    vel_new = vel + 0.5 * (a_n + a_new) * dt            # Update velocities using the average of the old and new accelerations

    for i in range(n):
        bodies[i].pos = pos_new[i]
        bodies[i].vel = vel_new[i]