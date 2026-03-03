"""
simulation.py

This module defines the Simulation class, which manages the state and evolution of a physics simulation involving multiple bodies.
The Simulation class handles the integration of the equations of motion using various numerical integrators, collision detection and resolution between bodies, and energy calculations for the system.
It also includes an adaptive time-stepping mechanism to ensure accuracy while maintaining computational efficiency.
The Simulation class provides methods to run the simulation for a specified duration and to log the state of the system at each time step for later analysis or visualization.
The Simulation class relies on the integrators defined in the integrators.py module and the mathematical functions defined in the math.py module to perform its operations.
It also uses NumPy for efficient numerical computations and array manipulations.

Functions:
------------
- __init__: Initialize the simulation with a list of bodies, an integrator, a force function, a time step, and a coefficient of restitution for collisions.
- step: Perform a single simulation step, including integration, collision handling, and state logging.
- run: Run the simulation for a specified total time.
- _pack_state: Convert the current state of the bodies into a flattened state vector for integration.
- _unpack_state: Update the bodies' positions and velocities from a given state vector.
- _compute_error: Compute the error of a proposed integration step for adaptive time stepping.
- _log_state: Log the current state of the system, including energies and trajectories, for later analysis or visualization.
- _handle_collisions: Check for and resolve collisions between bodies based on their positions and radii.
- _resolve_collision: Resolve a collision between two bodies using the coefficient of restitution and positional correction to prevent interpenetration.
- _compute_energy: Compute the total kinetic and potential energy of the system at the current state.
- _adaptive_step: Perform an adaptive time step using the specified integrator and error control mechanism to ensure accuracy while maintaining computational efficiency.
"""

import numpy as np


class Simulation:
    """
    Simulation class to manage the state and evolution of a physics simulation involving multiple bodies.
    This class handles the integration of the equations of motion using various numerical integrators, collision detection and resolution between bodies, and energy calculations for the system. It also includes an adaptive time-stepping mechanism to ensure accuracy while maintaining computational efficiency. The Simulation class provides methods to run the simulation for a specified duration and to log the state of the system at each time step for later analysis or visualization.
    The Simulation class relies on the integrators defined in the integrators.py module and the mathematical functions defined in the math.py module to perform its operations. It also uses NumPy for efficient numerical computations and array manipulations.

    Parameters:
    -----------
    bodies : list of Body
        A list of Body objects representing the bodies in the simulation, each with properties such as mass, position, velocity, and radius.
    integrator : function
        A numerical integration function (e.g., euler, rk4, velocity_verlet) that takes the current state and computes the next state based on the equations of motion.
    force_fn : function
        A function that computes the accelerations for the bodies based on their positions, typically representing the forces acting on the bodies (e.g., gravitational forces).
    dt : float
        The initial time step for the integration, which may be adjusted adaptively during the simulation.
    restitution : float, optional
        The coefficient of restitution for collisions between bodies, which determines how much kinetic energy is conserved during collisions (default is 0.8).
    """

    # Gravitational constant and small epsilon to avoid singularities
    G = 6.67430e-11
    EPS = 1e-10

    def __init__(self, bodies, integrator, force_fn, dt, restitution=0.8):
        self.bodies = bodies
        self.integrator = integrator
        self.force_fn = force_fn
        self.dt = dt
        self.t = 0
        self.r = restitution
        self.traj = [[] for _ in bodies]  # To store position history for each body
        self.vels = [[] for _ in bodies]  # To store velocity history for each body
        self.time = []  # To store time history for plotting
        self.energy_history = {  # To store energy history for plotting
            "kinetic": [],
            "potential": [],
            "total": [],
        }
        self.momentum_history = {
            "linear": [],  # To store linear momentum history for plotting
            "angular": [],  # To store angular momentum history for plotting
        }
        self.order_map = (
            {  # Map integrator functions to their order for adaptive time stepping
                "euler": 1,
                "rk4": 4,
                "velocity_verlet": 2,
            }
        )
        self.safety = 0.9  # Safety factor for adaptive time stepping
        self.atol = 1e-6  # Absolute tolerance for adaptive time stepping
        self.rtol = 1e-3  # Relative tolerance for adaptive time stepping
        self.max_dt = dt * 10  # Maximum allowed time step for adaptive time stepping
        self.min_dt = dt * 1e-6  # Minimum allowed time step for adaptive time stepping

        self.masses = np.array(
            [b.mass for b in self.bodies]
        )  # Precompute masses for efficiency
        self.pos = np.array(
            [b.pos for b in self.bodies]
        )  # Precompute positions for efficiency
        self.vel = np.array(
            [b.vel for b in self.bodies]
        )  # Precompute velocities for efficiency

        self._log_state()  # Log the initial state of the system

    def step(self):
        """
        Perform a single simulation step, including integration, collision handling, and state logging.
        This method uses the specified integrator to compute the next state of the system based on the current state and the forces acting on the bodies. It then updates the time,
        handles any collisions that may have occurred during the step, and logs the new state of the system for later analysis or visualization.
        The method relies on the _adaptive_step() method to perform the integration with adaptive time stepping, which ensures that the integration is accurate while maintaining computational efficiency.
        After the integration step, it calls the _handle_collisions() method to check for and resolve any collisions between bodies,
        and then calls the _log_state() method to record the current state of the system, including energies and trajectories, for later analysis or visualization.
        """
        new_state, used_dt = (
            self._adaptive_step()
        )  # Perform adaptive time step to get the new state and the actual time step used
        self._unpack_state(
            new_state
        )  # Update the bodies' positions and velocities from the new state vector
        self.pos = np.array(
            [b.pos for b in self.bodies]
        )  # Update precomputed positions after unpacking the new state
        self.vel = np.array(
            [b.vel for b in self.bodies]
        )  # Update precomputed velocities after unpacking the new state
        self.t += used_dt  # Update the current time by the actual time step used
        self._handle_collisions()  # Check for and resolve any collisions between bodies based on their updated positions
        self._log_state()  # Log the current state of the system, including energies and trajectories, for later analysis or visualization

    def run(self, T):
        """
        Run the simulation for a given time T.

        This method repeatedly calls the step() method until the total simulation time T is reached.
        It includes a check to adjust the final time step if the next step would exceed the total simulation time, ensuring that the simulation ends exactly at time T.
        The method uses a while loop to continue stepping through the simulation until the current time self.t reaches or exceeds the specified total time T.
        Inside the loop, it checks if the next step would exceed T, and if so, it temporarily adjusts the time step to ensure that the simulation ends exactly at time T. After the loop,
        the simulation will have evolved the system from the initial state to the state at time T, with all intermediate states logged for analysis or visualization.

        Parameters:
        -----------
        T : float
            The total time for which to run the simulation.The simulation will evolve the system from the initial state to the state at time T, with all intermediate states logged for analysis or visualization.
        """
        while self.t < T:
            if (
                self.t + self.dt > T
            ):  # Check if the next step would exceed the total simulation time
                old_dt = self.dt  # Store the original time step
                self.dt = T - self.t  # Adjust the time step to end exactly at time T
                self.step()
                self.dt = (
                    old_dt  # Restore the original time step for future steps if needed
                )
            else:  # If the next step does not exceed T, simply perform a regular step
                self.step()

    def _pack_state(self):
        """
        Convert the current state of the bodies into a flattened state vector for integration.
        This method takes the positions and velocities of all bodies in the simulation and concatenates them into a single flattened NumPy array.
        The first part of the array contains the positions of all bodies (flattened), and the second part contains the velocities of all bodies (also flattened).
        This state vector is used as input to the integrator functions to compute the next state of the system based on the equations of motion.
        The method iterates through the list of bodies, extracts their positions and velocities, and constructs the state vector in the required format for the integrators.
        The resulting state vector is a one-dimensional NumPy array that contains all the necessary information about the current state of the system for the integration step.

        Returns:
        --------
        np.ndarray
            A flattened array containing the positions and velocities of all bodies in the simulation, formatted for input to the integrator functions.
        """
        return np.concatenate(
            (
                np.array([b.pos for b in self.bodies]).flatten(),
                np.array([b.vel for b in self.bodies]).flatten(),
            )
        )

    def _unpack_state(self, state):
        """
        Update the bodies' positions and velocities from a given state vector.
        This method takes a flattened state vector (as produced by the _pack_state() method) and updates the positions and velocities of the bodies in the simulation accordingly.
        It extracts the positions and velocities from the state vector, reshapes them into the appropriate format, and assigns them back to the corresponding Body objects in the simulation.
        The method assumes that the state vector is structured such that the first part contains the positions of all bodies (flattened) and the second part contains the velocities of all bodies (also flattened).
        It uses the number of bodies in the simulation to correctly reshape the positions and velocities before updating the Body objects.

        Parameters:
        -----------
        state : np.ndarray
            A flattened array containing the positions and velocities of all bodies in the simulation, formatted as produced by the _pack_state() method. The method will extract the positions and velocities from this state vector and update the corresponding Body objects in the simulation with the new positions and velocities.

        Note: This method is used to update the state of the bodies in the simulation after an integration step has been performed, allowing the simulation to evolve over time based on the computed next state.
        The method iterates through the list of bodies, extracts the corresponding positions and velocities from the state vector, and assigns them back to the Body objects, effectively updating their state for the next iteration of the simulation.
        """
        n = len(self.bodies)
        pos = state[: n * 3].reshape(n, 3)
        vel = state[n * 3 :].reshape(n, 3)

        for i, b in enumerate(self.bodies):
            b.pos = pos[i]
            b.vel = vel[i]

    def _compute_error(self, state, dt):

        # One full step
        y_full = self.integrator(self.masses, state, self.force_fn, dt, self.t)

        # Two half step
        y_half = self.integrator(self.masses, state, self.force_fn, dt / 2, self.t)
        y_half = self.integrator(
            self.masses, y_half, self.force_fn, dt / 2, self.t + dt / 2
        )

        # Compute error using the difference between the full step and the two half steps, normalized by a combination of absolute and relative tolerances to ensure that the error is appropriately scaled for the magnitude of the solution.
        # The error is computed as the root mean square of the normalized error vector, which provides a single scalar value representing the overall error of the integration step.
        # This error value is then used in the adaptive time-stepping mechanism to adjust the time step for future steps, ensuring that the integration remains accurate while maintaining computational efficiency.
        scale = self.atol + self.rtol * np.abs(y_half)
        err_vec = (y_full - y_half) / scale
        error = np.sqrt(np.mean(err_vec**2))
        return error, y_half

    def _log_state(self):
        """
        Log the current state of the simulation, including energies and trajectories.
        This method computes the kinetic and potential energy of the system at the current state and stores these values in the energy history for later analysis or visualization.
        It also appends the current positions and velocities of each body to their respective trajectory and velocity histories, allowing for tracking the evolution of the system over time.
        Finally, it records the current time in the time history for plotting purposes.
        The method relies on the _compute_energy() method to calculate the kinetic and potential energy of the system based on the current positions and velocities of the bodies, and it uses the properties of the Body objects to access their positions and velocities for logging.
        This logged data can be used to analyze the behavior of the system, visualize trajectories, and plot energy changes over time to understand the dynamics of the simulation.
        """
        KE, PE = (
            self._compute_energy()
        )  # compute kinetic and potential energy of the system at the current state
        self.energy_history["kinetic"].append(KE)  # store Kinetic Energy
        self.energy_history["potential"].append(PE)  # store Potential Energy
        self.energy_history["total"].append(KE + PE)  # store Total Energy

        P, L = (
            self._compute_momenta()
        )  # compute linear and angular momentum of the system at the current state
        self.momentum_history["linear"].append(np.linalg.norm(P))   # store Linear momentum
        self.momentum_history["angular"].append(np.linalg.norm(L))  # store Angular momentum

        for i, body in enumerate(self.bodies):
            self.traj[i].append(body.pos.copy())  # store Position
            self.vels[i].append(body.vel.copy())  # store Velocity

        self.time.append(self.t)  # store Time

    def _handle_collisions(self):
        """
        Check for and resolve collisions between bodies based on their positions and radii.
        This method iterates through all unique pairs of bodies in the simulation, calculates the distance between
        their centers, and checks if they are colliding (i.e., if the distance is less than the sum of their radii).
        If a collision is detected, it calls the _resolve_collision() method to apply the appropriate response based on the coefficient of restitution and to ensure that the bodies do not interpenetrate after the
        collision is resolved.

        The method uses a nested loop to compare each body with every other body, but only checks each pair once to avoid redundant calculations.
        It also includes a check to skip collision resolution if the bodies are moving apart, as this indicates that they have already collided and are now separating.

        parameters:
        -----------
        None

        The method relies on the properties of the Body objects (such as position, velocity, mass, and radius) to determine the collision response and to update the state of the bodies accordingly.
        """
        n = len(self.bodies)

        for i in range(n):
            bi = self.bodies[i]
            for j in range(i + 1, n):
                bj = self.bodies[j]
                delta = bi.pos - bj.pos  # Vector from bj to bi
                dist = np.linalg.norm(delta)  # Distance between centers of bi and bj

                if (
                    dist < self.EPS
                ):  # Avoid singularity and extremely large forces when bodies are very close
                    continue

                if dist < bi.radius + bj.radius:  # Collision detected
                    self._resolve_collision(bi, bj, delta, dist)

    def _resolve_collision(self, bi, bj, delta, dist):
        """
        Resolve a collision between two bodies bi and bj using the coefficient of restitution and positional correction to prevent interpenetration.
        This method calculates the normal vector of the collision, determines the relative velocity of the bodies along
        the collision normal, and applies an impulse to the velocities of the bodies based on the coefficient of restitution.
        It also includes a positional correction step to ensure that the bodies do not interpenetrate after the collision is resolved, which can occur due to numerical errors or if the time step is large enough
        that the bodies overlap significantly before the collision is detected.

        Parameters:
        -----------
        bi : Body
            The first body involved in the collision
        bj : Body
            The second body involved in the collision
        delta : np.ndarray
            The vector from bj to bi (bi.pos - bj.pos)
        dist : float
            The distance between the centers of bi and bj

        The method first checks if the bodies are moving towards each other by calculating the relative velocity along the collision normal. If they are moving apart, it returns without applying any collision response.
        If they are moving towards each other, it calculates the impulse scalar based on the coefficient of restitution and the masses of the bodies, and applies this impulse to update their velocities.
        Finally, it computes the penetration depth and applies a positional correction to ensure that the bodies are no longer overlapping after the collision is resolved.
        """
        n_hat = delta / dist
        v_rel = bi.vel - bj.vel
        # Check if bodies are moving towards each other
        if np.dot(v_rel, n_hat) > 0:
            return  # No collision if they are moving apart
        # Compute impulse scalar
        inv_mass_sum = (1 / bi.mass) + (1 / bj.mass)
        J = -(1 + self.r) * np.dot(v_rel, n_hat) / inv_mass_sum
        # Apply impulse to the velocities of the bodies
        bi.vel += (J / bi.mass) * n_hat
        bj.vel -= (J / bj.mass) * n_hat

        # Positional correction to prevent interpenetration
        penetration = bi.radius + bj.radius - dist
        if penetration > 0:
            correction = penetration / inv_mass_sum
            bi.pos += (correction / bi.mass) * n_hat
            bj.pos -= (correction / bj.mass) * n_hat

    def _compute_energy(self):
        """
        Compute the total kinetic and potential energy of the system at the current state.
        This method calculates the kinetic energy (KE) of each body based on its mass and velocity, and the potential energy (PE) of the system based on the gravitational interactions between all unique pairs of bodies.
        The kinetic energy is computed using the formula KE = 0.5 * mass * velocity^2, while the potential energy is computed using the formula PE = -G * m1 * m2 / r for each pair of bodies, where G is the gravitational constant, m1 and m2 are the masses of the bodies, and r is the distance between their centers.
        The method includes a check to avoid singularities when bodies are very close to each other by ensuring that the distance used in the potential energy calculation is not below a small threshold (EPS).

        Parameters:
        -----------
        None

        The method iterates through all bodies to compute the kinetic energy and through all unique pairs of bodies to compute the potential energy, summing these values to return the total kinetic and potential energy of the system at the current state.

        Returns:
        --------
        KE : float
            The total kinetic energy of the system
        PE : float
            The total potential energy of the system
        """
        KE = 0.0
        PE = 0.0
        n = len(self.bodies)

        for i in range(n):
            bi = self.bodies[i]
            KE += 0.5 * bi.mass * np.dot(bi.vel, bi.vel)

            for j in range(i + 1, n):
                bj = self.bodies[j]
                r = np.linalg.norm(bi.pos - bj.pos)
                r = max(r, self.EPS)
                PE -= self.G * bi.mass * bj.mass / r
        return KE, PE

    def _adaptive_step(self):
        """
        Perform an adaptive time step using the specified integrator and error control mechanism to ensure accuracy while maintaining computational efficiency.
        This method uses the _compute_error() method to estimate the error of a proposed integration step and adjusts the time step accordingly based on the specified absolute and relative tolerances, as well as a safety factor to prevent overly aggressive time step changes.
        The method continues to adjust the time step until an acceptable error level is achieved, at which point it returns the new state of the system and the actual time step used for the integration.
        The method relies on the order of the integrator (as defined in the order_map) to determine how the error scales with the time step, allowing it to compute the appropriate factor for adjusting the time step based on the estimated error.
        This adaptive time-stepping mechanism helps to ensure that the integration remains accurate while avoiding unnecessarily small time steps that can lead to increased computational cost,
        especially in situations where the system's dynamics change rapidly or when bodies come close to each other, which can lead to large forces and rapid changes in velocities.

        Returns:
        --------
        new_state : np.ndarray
            The updated state vector after a successful integration step with an acceptable error level
        used_dt : float
            The actual time step used for the integration, which may be different from the initial time step due to the adaptive time-stepping mechanism

        The method uses a while loop to continuously compute the error and adjust the time step until an acceptable error level is achieved. Inside the loop, it computes the error using the _compute_error() method,
        calculates the factor for adjusting the time step based on the error and the order of the integrator, and then updates the time step while ensuring that it remains within specified minimum and maximum bounds.
        If the error is acceptable (i.e., less than or equal to 1), it returns the new state and the actual time step used; otherwise, it continues to adjust the time step and recompute the error until an acceptable level is reached.
        """
        state = self._pack_state()
        p = self.order_map[self.integrator.__name__]
        if p is None:  # Check if the integrator is recognized
            raise ValueError("Unknown integrator order")
        dt = self.dt  # Start with the current time step for the integration
        while True:
            error, y_new = self._compute_error(state, dt)
            error = max(
                error, 1e-14
            )  # Avoid division by zero in case of very small errors
            factor = self.safety * (1.0 / error) ** (
                1 / (p + 1)
            )  # Compute the factor for adjusting the time step based on the error and the order of the integrator
            new_dt = dt * factor
            new_dt = min(
                self.max_dt, max(self.min_dt, new_dt)
            )  # Ensure the new time step is within the specified bounds

            if (
                error <= 1.0
            ):  # If the error is acceptable, return the new state and the actual time step used
                self.dt = new_dt
                return y_new, dt
            else:  # If the error is not acceptable, update the time step and recompute the error
                dt = new_dt

    def _compute_momenta(self):
        """
        Compute the total linear and angular momentum of the system at the current state.
        This method calculates the linear momentum (P) of the system by summing the product of mass and velocity for all bodies,
        and the angular momentum (L) by summing the cross product of the position vector (relative to the center of mass)
        and the velocity vector (also relative to the center of mass) for all bodies, weighted by their masses.
        The method first computes the center of mass position (R_COM) and velocity (V_COM) of the system,
        then calculates the relative position (ri) and velocity (vi) of each body with respect to the center of mass,
        and finally computes the total linear and angular momentum based on these relative quantities.
        """
        R_COM = np.sum(self.masses[:, None] * self.pos, axis=0) / np.sum(self.masses)
        V_COM = np.sum(self.masses[:, None] * self.vel, axis=0) / np.sum(self.masses)

        P = np.sum(self.masses[:, None] * self.vel, axis=0)

        ri = self.pos - R_COM
        vi = self.vel - V_COM

        L = np.sum((self.masses[:, None] * np.cross(ri, vi)), axis=0)
        return P, L
