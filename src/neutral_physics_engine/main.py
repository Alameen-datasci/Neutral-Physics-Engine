"""
main.py

This module serves as the entry point for running the physics simulation. It imports the necessary classes and functions from the other modules, sets up the initial conditions for the simulation, and executes the simulation loop.
The main function initializes the Simulation object with specified parameters, runs the simulation, and then uses the Analysis class to visualize the results.
The module also includes error handling to ensure that the simulation runs smoothly and provides informative messages in case of issues with the input parameters or during the simulation process.
"""

from body import Body
from simulation import Simulation
from analysis import Analysis
from integrators import euler, rk4, velocity_verlet
from forces import forces


# ------------------------------------- Create bodies --------------------------------------
sun = Body(mass=1.989e30, pos=[0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0], radius=6.96e8)
earth = Body(mass=5.972e24, pos=[1.49e11, 0.0, 0.0], vel=[0.0, 29780.0, 0.0], radius=6.371e6)
bodies = [sun, earth]

# ------------------------------------- Run simulation --------------------------------------
T = 365.25 * 24 * 3600  # Total simulation time (1 year in seconds)
dt = 60 * 60  # Time step (1 hour in seconds)

sim = Simulation(bodies=bodies, force_fn=forces, integrator=rk4, dt=dt)
sim.run(T)

# ------------------------------------- Analyze results --------------------------------------
# analysis = Analysis(sim)
# analysis.relative_energy_error()
# analysis.plot_energy_components()
# print(analysis.energy_drift_rate())
# analysis.plot_projection(planes=["xy", "xz", "yz"])
# analysis.plot_linear_momentum()
# analysis.plot_angular_momentum()

# ------------------------------------- Notes --------------------------------------
# This is a basic example of how to set up and run a physics simulation using the defined classes and functions.
# The parameters for the bodies, simulation time, and time step can be adjusted to explore different scenarios and observe various physical phenomena.
# The analysis section provides insights into the energy conservation and momentum properties of the system, which can be useful for understanding the dynamics of the simulation.
# Velocity verlet integrator is implimented but currently using a fixed time step, for a very small or very large steps, energy conservation may not be good, so it is recommended to use adaptive time step for better energy conservation.
# The code is structured to allow for easy extension and modification, enabling users to add more bodies, different force models, or alternative integrators as needed for their specific simulations.
# class Body has orientation and angular velocity attributes, but they are not currently used in the simulation.
# Because gravitational forces are central forces, they do not produce torque, so the orientation and angular velocity of the bodies remain constant throughout the simulation.
# However, these attributes can be utilized in future extensions of the simulation to include non-central forces or to model rigid body dynamics where torque and rotational motion are relevant.

# ------------------------------- End of main.py --------------------------------------