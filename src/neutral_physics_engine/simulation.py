"""
simulation.py

This module provides the core orchestration engine for the N-body physics simulation.
It acts as the central hub, tying together numerical integration, spatial
partitioning (gravity fields), collision detection, and high-performance data logging.

The Simulation class utilizes adaptive time-stepping based on local truncation error
estimates (via step doubling/Richardson extrapolation) to maintain strict error bounds
while maximizing computational throughput. It treats the entire N-body system as a
single composite phase-space vector to optimize NumPy operations during ODE integration.

Key features:
- Adaptive time-stepping with absolute and relative tolerance control.
- Phase-space vectorization for high-performance ODE integration.
- Exact-time simulation stopping bounds.
- Invariant tracking (Energy and Momenta calculations relative to the Center of Mass).
- Seamless delegation to CollisionSystem and HDF5Writer subsystems.

Dependencies:
    numpy : Matrix and vector mathematics
    neutral_physics_engine.collision : Rigid-body impact resolution
    neutral_physics_engine.io : Disk-buffered HDF5 telemetry logging
"""

from neutral_physics_engine.collision import CollisionSystem
from neutral_physics_engine.io import HDF5Writer
import numpy as np


class Simulation:
    """
    The primary orchestrator for the N-body physics engine.

    Manages the time evolution of a system of bodies under mutual gravitational
    attraction and physical collisions. It guarantees numerical stability by
    dynamically adjusting the integration time step (dt) based on the local
    truncation error of the chosen integration scheme.
    """

    # Gravitational constant and small epsilon to avoid singularities
    G = 6.67430e-11
    EPS = 1e-10

    def __init__(
        self,
        bodies: list,
        integrator,
        field,
        dt: int,
        restitution: int = 0.8,
        hdf5_writer: HDF5Writer | None = None,
    ):
        """
        Initialize the simulation environment.

        Parameters:
        -----------
        bodies : list
            List of Body objects defining the initial state of the system.
        integrator : callable
            Numerical ODE solver (e.g., euler, rk4, velocity_verlet).
        field : GravityField
            The gravitational field compute instance (utilizing the Octree).
        dt : float
            The initial proposed time step in seconds.
        restitution : float, optional
            Coefficient of restitution for collisions [0.0, 1.0] (default 0.8).
        hdf5_writer : HDF5Writer | None, optional
            Telemetry logger for writing phase-space data to disk.
        """
        self.bodies = bodies
        self.integrator = integrator
        self.field = field
        self.dt = dt
        self.t = 0
        self.restitution = restitution
        self.kinetic = 0.0
        self.potential = 0.0
        self.total = 0.0
        self.linear_p = 0.0
        self.angular_p = 0.0
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
        self.masses = np.array([b.mass for b in self.bodies])  # Precompute masses for efficiency
        self.pos = np.array([b.pos for b in self.bodies])  # Precompute positions for efficiency
        self.vel = np.array([b.vel for b in self.bodies])  # Precompute velocities for efficiency
        self.hdf5_writer = hdf5_writer
        self._log_state()  # Log the initial state of the system

    def step(self) -> None:
        """
        Advance the simulation by a single adaptive time step.

        This process involves:
        1. Determining the optimal step size and calculating the new kinematic state.
        2. Unpacking the phase-space vector back to individual bodies.
        3. Detecting and resolving spatial overlaps (collisions).
        4. Computing diagnostics (energies, momenta) and logging the state.
        """
        # Perform adaptive time step to get the new state and the actual time step used
        new_state, used_dt = self._adaptive_step()
        # Update the bodies' positions and velocities from the new state vector
        self._unpack_state(new_state)
        # Update precomputed positions after unpacking the new state
        self.pos = np.array([b.pos for b in self.bodies])
        # Update precomputed velocities after unpacking the new state
        self.vel = np.array([b.vel for b in self.bodies])
        self.t += used_dt  # Update the current time by the actual time step used
        self._handle_collisions()  # Check for and resolve any collisions between bodies based on their updated positions
        self._log_state()  # Log the current state of the system, including energies and trajectories, for later analysis or visualization

    def run(self, T: np.float64) -> None:
        """
        Run the simulation until the specified system time is reached.

        Automatically truncates the final time step to ensure the simulation ends
        exactly at time T, preventing over-integration.

        Parameters:
        -----------
        T : np.float64
            The target total simulation time (in seconds).
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

    def _pack_state(self) -> np.ndarray:
        """
        Flatten the system's kinematic variables into a single 1D phase-space vector.

        Standard ODE solvers operate on a contiguous state vector. This concatenates
        all 3D positions followed by all 3D velocities.

        Returns:
        --------
        np.ndarray
            1D array of shape (N*6,) representing the full system state.
        """
        return np.concatenate(
            (
                np.array([b.pos for b in self.bodies]).flatten(),
                np.array([b.vel for b in self.bodies]).flatten(),
            )
        )

    def _unpack_state(self, state: np.ndarray) -> None:
        """
        Distribute a flattened phase-space vector back into the system's Body objects.

        Parameters:
        -----------
        state : np.ndarray
            1D array of shape (N*6,) containing concatenated positions and velocities.
        """
        n = len(self.bodies)
        pos = state[: n * 3].reshape(n, 3)
        vel = state[n * 3 :].reshape(n, 3)

        for i, b in enumerate(self.bodies):
            b.pos = pos[i]
            b.vel = vel[i]

    def _compute_error(self, state: np.ndarray, dt: np.float64):
        """
        Estimate the local truncation error of the current time step using
        Richardson extrapolation (step doubling).

        Integrates the system over `dt` once, and over `dt/2` twice. The difference
        between the two solutions yields an asymptotic estimate of the integration error.

        Parameters:
        -----------
        state : np.ndarray
            The current packed phase-space vector.
        dt : np.float64
            The proposed time step.

        Returns:
        --------
        tuple[float, np.ndarray]
            A tuple containing the normalized scalar error magnitude and the
            higher-accuracy state vector (`y_half`).
        """
        # One full step
        y_full = self.integrator(self.masses, state, self.field, dt, self.t)

        # Two half step
        y_half = self.integrator(self.masses, state, self.field, dt / 2, self.t)
        y_half = self.integrator(
            self.masses, y_half, self.field, dt / 2, self.t + dt / 2
        )

        # Compute error using the difference between the full step and the two half steps, normalized by a combination of absolute and relative tolerances to ensure that the error is appropriately scaled for the magnitude of the solution.
        # The error is computed as the root mean square of the normalized error vector, which provides a single scalar value representing the overall error of the integration step.
        # This error value is then used in the adaptive time-stepping mechanism to adjust the time step for future steps, ensuring that the integration remains accurate while maintaining computational efficiency.
        scale = self.atol + self.rtol * np.abs(y_half)
        err_vec = (y_full - y_half) / scale
        error = np.sqrt(np.mean(err_vec**2))
        return error, y_half

    def _log_state(self) -> None:
        """
        Compute global invariants (energy, momentum) and push the current
        simulation frame to the I/O buffer.
        """
        KE, PE = self._compute_energy()
        self.kinetic = KE
        self.potential = PE
        self.total = KE + PE

        P, L = self._compute_momenta()
        self.linear_p = float(np.linalg.norm(P))
        self.angular_p = float(np.linalg.norm(L))

        if self.hdf5_writer is not None:
            self.hdf5_writer.append_step(
                t=self.t,
                positions=self.pos,
                velocities=self.vel,
                kinetic=self.kinetic,
                potential=self.potential,
                total=self.total,
                linear_p=self.linear_p,
                angular_p=self.angular_p,
            )

    def _handle_collisions(self) -> None:
        """
        Delegate spatial intersection testing and impulse resolution to the
        CollisionSystem, updating the internal state arrays post-resolution.
        """
        # extracting parameters for collision
        masses = self.masses
        positions = self.pos
        velocities = self.vel
        radii = np.array([b.radius for b in self.bodies])
        # resolving collisions
        collision = CollisionSystem(
            masses, positions, velocities, radii, self.restitution
        )
        collision.resolve_collisions()
        # updating position and velocity
        for k, b in enumerate(self.bodies):
            b.pos = positions[k]
            b.vel = velocities[k]

        self.pos = positions
        self.vel = velocities

    def _compute_energy(self) -> tuple[float, float]:
        """
        Calculate the macroscopic kinetic and potential energies of the system.

        Returns:
        --------
        tuple[float, float]
            (Total Kinetic Energy, Total Potential Energy) in Joules.
        """
        KE = 0.5 * np.sum(self.masses * np.sum(self.vel**2, axis=1))
        PE = self.field.compute_potential(self.masses, self.pos)
        return KE, PE

    def _adaptive_step(self):
        """
        Determine and apply the optimal time step satisfying error tolerances.

        Uses an established PID-like control algorithm for step size prediction.
        If a proposed step results in an error > 1.0, the step is rejected, `dt`
        is shrunken, and the calculation is repeated until convergence.

        Raises:
        -------
        ValueError
            If the assigned integrator order is not defined in `self.order_map`.

        Returns:
        --------
        tuple[np.ndarray, float]
            The accepted phase-space state vector and the `dt` value used to reach it.
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

    def _compute_momenta(self) -> tuple[np.ndarray, np.ndarray]:
        """
        Calculate total linear momentum and internal angular momentum.

        Angular momentum is calculated relative to the system's barycenter (Center
        of Mass) to decouple it from translational motion.

        Returns:
        --------
        tuple[np.ndarray, np.ndarray]
            (Total Linear Momentum vector, Total Angular Momentum vector).
        """
        R_COM = np.sum(self.masses[:, None] * self.pos, axis=0) / np.sum(self.masses)
        V_COM = np.sum(self.masses[:, None] * self.vel, axis=0) / np.sum(self.masses)

        P = np.sum(self.masses[:, None] * self.vel, axis=0)

        ri = self.pos - R_COM
        vi = self.vel - V_COM

        L = np.sum((self.masses[:, None] * np.cross(ri, vi)), axis=0)
        return P, L
