"""
analysis.py

This module defines the Analysis class, which provides methods for analyzing the results of a physics simulation.
The Analysis class takes a Simulation object as input and uses its data to perform various analyses, such as calculating energy conservation, momentum conservation, and trajectory projections.
The class includes methods to plot the relative energy error, energy components, trajectory projections, and the magnitude of linear and angular momentum over time.
"""

import numpy as np
import matplotlib.pyplot as plt

class Analysis:
    """
    A class for analyzing the results of a physics simulation.

    This class provides methods to calculate and visualize various aspects of the simulation, such as energy conservation, momentum conservation, and trajectory projections. It takes a Simulation object as input and uses its data to perform the analysis.

    Attributes:
        sim (Simulation): The simulation object containing the data to be analyzed.

    Methods:
        relative_energy_error(): Calculate and plot the relative energy error over time.
        plot_energy_components(): Plot the kinetic, potential, and total energy over time.
        energy_drift_rate(): Calculate the rate of energy drift over the course of the simulation.
        plot_trajectory_3d(): Plot the 3D trajectories of the bodies in the simulation (not yet implemented).
        plot_projection(planes, body_index=None): Plot the trajectory projections onto specified planes (xy, xz, yz) for selected bodies.
        plot_linear_momentum(): Plot the magnitude of the total linear momentum over time.
        plot_angular_momentum(): Plot the magnitude of the total angular momentum over time.
    """
    def __init__(self, sim):
        self.sim = sim
        if self.sim is None:
            raise ValueError("Simulation object is None. Please provide a valid simulation instance.")
        if len(self.sim.time) == 0:
            raise ValueError("Simulation has no time data.")

    def relative_energy_error(self):
        """
        Calculate and plot the relative energy error over time.

        The relative energy error is defined as (E(t) - E(0)) / E(0), where E(t) is the total energy at time t and E(0) is the initial total energy.
        """
        total_energy = self.sim.energy_history["total"]
        time = self.sim.time
        E0 = total_energy[0]
        relative_error = [(E - E0) / E0 for E in total_energy]

        fig, ax = plt.subplots()
        ax.plot(time, relative_error)
        ax.set_title(f"Relative Energy Error ({self.sim.integrator.__name__})")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Relative Energy Error")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)
        plt.show()

    def plot_energy_components(self):
        """
        Plot the kinetic, potential, and total energy over time.
        """
        fig, axs = plt.subplots(3, 1, figsize=(10, 8))
        fig.suptitle(f"Energy Components (T={self.sim.t:.1f}s, {self.sim.integrator.__name__})")
        
        axs[0].plot(self.sim.time, self.sim.energy_history["kinetic"])
        axs[0].set_title("Total Kinetic Energy")
        axs[0].set_xlabel("Time (s)")
        axs[0].set_ylabel("Energy (J)")
        axs[0].spines["right"].set_visible(False)
        axs[0].spines["top"].set_visible(False)

        axs[1].plot(self.sim.time, self.sim.energy_history["potential"])
        axs[1].set_title("Total Potential Energy")
        axs[1].set_xlabel("Time (s)")
        axs[1].set_ylabel("Energy (J)")
        axs[1].spines["right"].set_visible(False)
        axs[1].spines["top"].set_visible(False)

        axs[2].plot(self.sim.time, self.sim.energy_history["total"])
        axs[2].set_title("Total Energy (Kinetic + Potential)")
        axs[2].set_xlabel("Time (s)")
        axs[2].set_ylabel("Energy (J)")
        axs[2].spines["right"].set_visible(False)
        axs[2].spines["top"].set_visible(False)

        plt.tight_layout()
        plt.show()

    def energy_drift_rate(self):
        """
        Calculate the rate of energy drift over the course of the simulation.

        The energy drift rate is defined as (E_final - E_initial) / (E_initial * T), where E_final is the total energy at the end of the simulation, E_initial is the total energy at the start, and T is the total simulation time.
        """
        Ef = self.sim.energy_history["total"][-1]
        E0 = self.sim.energy_history["total"][0]
        T = self.sim.t
        return (Ef - E0) / (E0 * T)

    def plot_trajectory_3d(self):
        raise NotImplementedError("3D trajectory plotting is not yet implemented")

    def plot_projection(self, planes, body_index=None):
        """
        Plot the trajectory projections onto specified planes (xy, xz, yz) for selected bodies.

        Args:
            planes (list of str): A list of plane identifiers to plot. Valid options are "xy", "xz", and "yz".
            body_index (int or list of int, optional): The index or indices of the body/bodies to plot. If None, all bodies will be plotted. Defaults to None.
        """
        valid_planes = {
            "xy": (0, 1, "X Position (m)", "Y Position (m)"),
            "xz": (0, 2, "X Position (m)", "Z Position (m)"),
            "yz": (1, 2, "Y Position (m)", "Z Position (m)")
            }

        planes = [p.lower().strip() for p in planes if p.lower().strip() in valid_planes]
        if not planes:
            print("No valid planes selected for projection plotting.")
            return

        n = len(planes)

        fig, axs = plt.subplots(1, n, figsize=(10 * n, 8))
        if n == 1:
            axs = [axs]

        fig.suptitle(f"Trajectory Projections (T={self.sim.t:.1f}s)", fontsize=16)
        if isinstance(body_index, int):
            target_indices = [body_index]
        elif isinstance(body_index, (list, tuple)):
            target_indices = body_index
        else:
            target_indices = None

        for ax, plane in zip(axs, planes):
            idx_h, idx_v, label_h, label_v = valid_planes[plane]

            for i, body_traj in enumerate(self.sim.traj):
                if target_indices is not None and i not in target_indices:
                    continue
                coords = np.array(body_traj)

                if len(coords) > 0:
                    h_vals = coords[:, idx_h]
                    v_vals = coords[:, idx_v]

                    line, = ax.plot(h_vals, v_vals, label=f"Body {i}")
                    ax.scatter(h_vals[0], v_vals[0], color=line.get_color(), marker='o', s=40, label='_nolegend_')
                    ax.scatter(h_vals[-1], v_vals[-1], color=line.get_color(), marker='x', s=40, label='_nolegend_')

                    ax.set_title(f"{plane.upper()} Plane")
                    ax.set_xlabel(label_h)
                    ax.set_ylabel(label_v)
                    ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_linear_momentum(self):
        """
        Plot the magnitude of the linear momentum over time.

        The linear momentum magnitude is calculated as the Euclidean norm of the total linear momentum vector at each time step. This plot can help visualize how well linear momentum is conserved throughout the simulation.
        """
        fig, ax = plt.subplots()
        ax.plot(self.sim.time, self.sim.momentum_history["linear"])
        ax.set_title(f"Linear Momentum Magnitude (T={self.sim.t:.1f}s)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Linear Momentum (kg·m/s)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.tight_layout()
        plt.show()

    def plot_angular_momentum(self):
        """
        Plot the magnitude of the angular momentum over time.

        The angular momentum magnitude is calculated as the Euclidean norm of the total angular momentum vector at each time step. This plot can help visualize how well angular momentum is conserved throughout the simulation.
        """
        fig, ax = plt.subplots()
        ax.plot(self.sim.time, self.sim.momentum_history["angular"])
        ax.set_title(f"Angular Momentum Magnitude (T={self.sim.t:.1f}s)")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Angular Momentum (kg·m²/s)")
        ax.spines["right"].set_visible(False)
        ax.spines["top"].set_visible(False)

        plt.tight_layout()
        plt.show()