from neutral_physics_engine.body import Body
from neutral_physics_engine.simulation import Simulation
from neutral_physics_engine.io import HDF5Writer
from neutral_physics_engine.integrators import velocity_verlet
from neutral_physics_engine.gravity_field import GravityField

# ================================ Inner Solar System Simulation ================================
sun = Body(pos = [0.0, 0.0, 0.0], vel = [0.0, 0.0, 0.0], mass = 1.989e30, radius = 6.9634e8)
mercury = Body(pos = [5.79e10, 0.0, 0.0], vel = [0.0, 47400.0, 0.0], mass = 3.301e23, radius = 2.44e6)
venus = Body(pos = [1.082e11, 0.0, 0.0], vel = [0.0, 35000.0, 0.0], mass = 4.867e24, radius = 6.052e6)
earth = Body(pos = [1.496e11, 0.0, 0.0], vel = [0.0, 29780.0, 0.0], mass = 5.972e24, radius = 6.371e6)
mars = Body(pos = [2.279e11, 0.0, 0.0], vel = [0.0, 24100.0, 0.0], mass = 6.417e23, radius = 3.389e6)
bodies = [sun, mercury, venus, earth, mars]

# --- Simulation parameters and HDF5 writer setup ---
metadata = {
    "main": "inner_solar_system.py",
    "integrator": "euler",
    "adaptive_step": True,
    "units_length": "meter",
    "units_mass": "kilogram",
    "units_time": "second"
}

# -- Simulation parameters ---
T = 365.25 * 24 * 3600
dt = 60 * 60

# -- HDF5 writer setup and simulation execution ---
with HDF5Writer(
    filename="results/simulation_testing_4.h5",         # Output file path
    n_bodies=len(bodies),
    buffer_size=1000,                                   # Number of time steps to buffer before writing to disk
    compression="gzip",                                 # Compression algorithm (e.g., "gzip", "lzf", "szip")
    compression_opts=4,                                 # Compression level (e.g., 1-9 for gzip)
    metadata=metadata                                   # Additional metadata to store in the HDF5 file
) as writer:

    gravity = GravityField(theta=0.5)                   # Create a gravity field with a softening parameter (theta) to prevent singularities
    sim = Simulation(bodies=bodies, field=gravity, integrator=velocity_verlet, dt=dt, hdf5_writer=writer)
    sim.run(T)

# The simulation will run for one year (T) with a time step of one hour (dt). The positions, velocities, and other relevant data of the bodies will be stored in the specified HDF5 file for later analysis and visualization.
# Note: The inner solar system will exhibit complex orbital dynamics due to the gravitational interactions between the sun and the planets. You can analyze the resulting HDF5 file to study the trajectories and interactions of the planets over time.
# you can use analysis.py to visualize the results of the simulation.

# ================================ End of Inner Solar System Simulation ================================