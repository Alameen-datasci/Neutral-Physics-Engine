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
    Manages the physics simulation of a system of bodies.

    Parameters:
    -----------
    bodies : list of Body
        List of Body objects in the simulation
    integrator : function
        Numerical integration function to update body states (e.g., rk4_step)
    force_fn : function
        Function that computes the forces (accelerations) on the bodies
    dt : float
        Time step for the simulation
    restitution : float
        Coefficient of restitution for collision handling (default: 0.8)

    The Simulation class provides methods to step through the simulation, run it for a specified duration, and plot the results.
    It also handles collision resolution between bodies and tracks the kinetic, potential, and total energy of
    the system over time.

    Methods:
    --------
    step() : Advances the simulation by one time step, updating body states and handling collisions.
    run(T) : Runs the simulation for a total time T, calling step() iteratively.
    plot() : Generates plots of speed vs time and energy vs time for the simulation results.

    Internal Methods:
    -----------------
    _resolve_collision(bi, bj) : Resolves a collision between two bodies bi and bj using the coefficient of restitution and positional correction to prevent interpenetration.
    """
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
        This method updates the positions and velocities of all bodies using the specified integrator and force function.
        It also computes the kinetic and potential energy of the system, handles collisions between bodies, and
        records the trajectory and energy history for analysis and plotting.

        The method iterates over all unique pairs of bodies to check for collisions and resolve them if necessary.
        It then updates the positions and velocities of the bodies based on the computed accelerations from the force function.
        Finally, it records the current state of the system for later visualization.
        """
        # Update positions and velocities using the integrator and force function
        self.integrator(self.bodies, self.force_fn, self.dt, self.t)

        n = len(self.bodies)
        G = 6.67430e-11
        EPS = 1e-10
        KE = 0
        PE = 0

        for i in range(n):
            for j in range(i+1, n):
                distance = np.linalg.norm(self.bodies[i].pos - self.bodies[j].pos)
                # Avoid singularity and extremely large forces when bodies are very close
                if distance < EPS:
                    distance = EPS
                # Check for collision and resolve it if necessary
                if distance < self.bodies[i].radius + self.bodies[j].radius:
                    self._resolve_collision(self.bodies[i], self.bodies[j])

                # Calculate potential energy contribution from this pair of bodies
                PE += -G * self.bodies[i].mass * self.bodies[j].mass / distance

            # Record trajectory and velocity for plotting
            self.traj[i].append(self.bodies[i].pos.copy())
            self.vels[i].append(self.bodies[i].vel.copy())
            KE += 0.5 * self.bodies[i].mass * np.dot(self.bodies[i].vel, self.bodies[i].vel)

        self.energy_history["kinetic"].append(KE)
        self.energy_history["potential"].append(PE)
        self.energy_history["total"].append(KE + PE)
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

    def _resolve_collision(self, bi, bj):
            """
            Resolve a collision between two bodies bi and bj using the coefficient of restitution and positional correction to prevent interpenetration.

            Parameters:
            -----------
            bi : Body
                First body involved in the collision
            bj : Body
                Second body involved in the collision

            The method calculates the normal vector between the two bodies, checks if they are moving towards each other,
            and if so, applies an impulse to their velocities based on the coefficient of restitution.
            It also applies a positional correction to ensure that the bodies do not interpenetrate after the
            collision resolution.

            The collision resolution is based on the principles of conservation of momentum and energy,
            modified by the coefficient of restitution to account for inelastic collisions.
            The positional correction is applied to prevent the bodies from sticking together or passing through each other after the collision is resolved.
            """
            n = bi.pos - bj.pos
            dist = np.linalg.norm(n)
            n_hat = n / dist
            v_rel = bi.vel - bj.vel
            # Check if bodies are moving towards each other
            if np.dot(v_rel, n_hat) > 0:
                 return       # No collision if they are moving apart
            # Compute impulse scalar
            mass_red = (1 / bi.mass) + (1 / bj.mass)
            J = - (1 + self.r) * np.dot(v_rel, n_hat) / mass_red
            # Apply impulse to the velocities of the bodies
            bi.vel += (J / bi.mass) * n_hat
            bj.vel -= (J / bj.mass) * n_hat

            # Positional correction to prevent interpenetration
            penetration = bi.radius + bj.radius - dist
            if penetration > 0:
                 correction = penetration / mass_red
                 bi.pos += (correction / bi.mass) * n_hat
                 bj.pos -= (correction / bj.mass) * n_hat

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