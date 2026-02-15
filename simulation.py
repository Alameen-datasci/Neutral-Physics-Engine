"""
simulation.py

Defines the Simulation class which manages the state and execution of the physics simulation.

The Simulation class is responsible for stepping through time,
applying the integrator to update the positions and velocities of the bodies, handling collisions,
and recording the trajectory and energy history for analysis and plotting.
"""

import numpy as np
import matplotlib.pyplot as plt

class Simulation:
    """
    Manages the physics simulation of multiple bodies.

    Parameters:
    -----------
    bodies : list of Body
        List of Body objects representing the physical bodies in the simulation.
    integrator : function
        A function that updates the positions and velocities of the bodies based on the forces acting on them and the time step.
    force_fn : function
        A function that computes the accelerations for the bodies based on their current positions and masses.
    dt : float
        Time step for the simulation (in seconds)
    restitution : float
        Coefficient of restitution for collision handling (default: 0.8)

    The Simulation class provides methods to step through the simulation,
    run it for a specified total time, handle collisions between bodies,
    compute the kinetic and potential energy of the system, and generate plots of the results.

    methods:
    --------
    step() : Advances the simulation by one time step, updating positions and velocities, handling collisions, and recording energy and trajectory history.
    run(T) : Runs the simulation for a total time
    plot() : Generates plots of speed vs time and energy vs time for the simulation results.

    internal methods:
    -------------------
    _handle_collisions() : Checks for and resolves collisions between bodies based on their positions and radii.
    _resolve_collision(bi, bj, delta, dist) : Resolves a collision between two bodies bi and bj using the coefficient of restitution and positional correction to prevent interpenetration.
    _compute_energy() : Computes the total kinetic and potential energy of the system at the current state.
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
        self.traj = [[] for _ in bodies] # To store position history for each body
        self.vels = [[] for _ in bodies] # To store velocity history for each body
        self.time = []                   # To store time history for plotting
        self.energy_history = {          # To store energy history for plotting
            "kinetic" : [],
            "potential" : [],
            "total" : []
        }

    def step(self):
        """
        Advance the simulation by one time step.
        This method updates the positions and velocities of the bodies using the specified integrator and force function
        It also handles collisions between bodies, computes the kinetic and potential energy of the system, and records the trajectory and energy history for analysis and plotting.

        The method first calls the integrator to update the state of the bodies based on the forces acting on them.
        Then it checks for collisions between bodies and resolves them using the specified coefficient of restitution.
        Finally, it computes the kinetic and potential energy of the system, records the current state of the bodies, and advances the simulation time by the specified time step.
        """
        # Update positions and velocities using the integrator and force function
        self.integrator(self.bodies, self.force_fn, self.dt, self.t)

        # Handle collisions between bodies
        self._handle_collisions()

        # Compute and record energy and trajectory history
        KE, PE = self._compute_energy()
        self.energy_history["kinetic"].append(KE)
        self.energy_history["potential"].append(PE)
        self.energy_history["total"].append(KE + PE)
        
        for i, body in enumerate(self.bodies):
            self.traj[i].append(body.pos.copy())
            self.vels[i].append(body.vel.copy())

        self.t += self.dt
        self.time.append(self.t)

    def run(self, T):
        """
        Run the simulation for a total time T.
        This method repeatedly calls the step() method until the total simulation time T is reached.
        It ensures that the simulation advances in increments of the specified time step dt and that all state
        updates, collision handling, and energy calculations are performed at each step.

        Parameters:
        -----------
        T : float
            Total time to run the simulation (in seconds)
        
        The method calculates the number of steps needed to reach the total time T based on the time step dt and iteratively calls step() to advance the simulation.
        """
        for _ in range(int(T / self.dt)):
            self.step()

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
            for j in range(i+1, n):
                bj = self.bodies[j]
                delta = bi.pos - bj.pos                 # Vector from bj to bi
                dist = np.linalg.norm(delta)            # Distance between centers of bi and bj

                if dist < self.EPS:                     # Avoid singularity and extremely large forces when bodies are very close
                    continue

                if dist < bi.radius + bj.radius:        # Collision detected
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
            return       # No collision if they are moving apart
        # Compute impulse scalar
        inv_mass_sum = (1 / bi.mass) + (1 / bj.mass)
        J = - (1 + self.r) * np.dot(v_rel, n_hat) / inv_mass_sum
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

            for j in range(i+1, n):
                 bj = self.bodies[j]
                 r = np.linalg.norm(bi.pos - bj.pos)
                 r = max(r, self.EPS)
                 PE -= self.G * bi.mass * bj.mass / r
        return KE, PE

    def plot(self):
            """
            Generate plots of speed vs time and energy vs time for the simulation results.
            This method creates a 2x2 grid of subplots to visualize the speed of each body over time, as well as the kinetic energy, potential energy, and total energy of the system
            over time. It uses Matplotlib to create the plots and includes titles, axis labels, legends, and grid lines
            for better readability.

            The speed vs time plot shows how the speed of each body changes throughout the simulation, while the energy plots provide insight into the conservation of energy and the dynamics of the system.

            The method ensures that the layout of the plots is adjusted to accommodate the overall title and that the visualizations are clear and informative for analyzing the results of the simulation.

                Parameters:
                -----------
                None

            The method uses the recorded trajectory, velocity, and energy history from the simulation to generate the plots.
            """
            fig, axs = plt.subplots(2, 2, figsize=(12, 10))
            fig.suptitle(f"N-Body Simulation Results (T={self.t:.1f}s)", fontsize=16)

            # --- Top-Left: Speed vs Time ---
            # self.vels contains vectors [vx, vy, vz]. We compute the norm for speed.
            ax_speed = axs[0, 0]
            for i, body_vels in enumerate(self.vels):
                # Calculate speed (magnitude of velocity vector) for each step
                speeds = [np.linalg.norm(v) for v in body_vels]
                ax_speed.plot(self.time, speeds, label=f'Body {i}')
            
            ax_speed.set_title("Speed vs Time")
            ax_speed.set_xlabel("Time (s)")
            ax_speed.set_ylabel("Speed (m/s)")
            ax_speed.spines["right"].set_visible(False)
            ax_speed.spines["top"].set_visible(False)
            ax_speed.legend()
            ax_speed.grid(True, alpha=0.3)

            # --- Top-Right: Kinetic Energy vs Time ---
            ax_ke = axs[0, 1]
            ax_ke.plot(self.time, self.energy_history["kinetic"], color='tab:orange')
            ax_ke.set_title("Total Kinetic Energy")
            ax_ke.set_xlabel("Time (s)")
            ax_ke.set_ylabel("Energy (J)")
            ax_ke.spines["right"].set_visible(False)
            ax_ke.spines["top"].set_visible(False)
            ax_ke.grid(True, alpha=0.3)

            # --- Bottom-Left: Potential Energy vs Time ---
            ax_pe = axs[1, 0]
            ax_pe.plot(self.time, self.energy_history["potential"], color='tab:green')
            ax_pe.set_title("Total Potential Energy")
            ax_pe.set_xlabel("Time (s)")
            ax_pe.set_ylabel("Energy (J)")
            ax_pe.spines["right"].set_visible(False)
            ax_pe.spines["top"].set_visible(False)
            ax_pe.grid(True, alpha=0.3)

            # --- Bottom-Right: Total Energy vs Time ---
            ax_total = axs[1, 1]
            ax_total.plot(self.time, self.energy_history["total"], color='tab:red')
            ax_total.set_title("Total Energy (Kinetic + Potential)")
            ax_total.set_xlabel("Time (s)")
            ax_total.set_ylabel("Energy (J)")
            ax_total.spines["right"].set_visible(False)
            ax_total.spines["top"].set_visible(False)
            ax_total.grid(True, alpha=0.3)

            plt.tight_layout(rect=[0, 0.03, 1, 0.95]) # Adjust layout to make room for suptitle
            plt.show()