from neutral_physics_engine.body import Body
from neutral_physics_engine.simulation import Simulation
from neutral_physics_engine.io import HDF5Writer
from neutral_physics_engine.integrators import velocity_verlet
from neutral_physics_engine.gravity_field import GravityField

# ================================ Binary Star System Simulation ================================
star_A = Body(pos = [-7.5e10, 0.0, 0.0], vel = [0.0, -15000.0, 0.0], mass = 1.989e30, radius = 6.9634e8)
star_B = Body(pos = [7.5e10, 0.0, 0.0], vel = [0.0, 15000.0, 0.0], mass = 1.989e30, radius = 6.9634e8)
# planet = Body(pos = [0.0, 3.0e11, 0.0], vel = [-21000.0, 0.0, 0.0], mass = 5.972e24, radius = 6.371e6)
bodies = [star_A, star_B]

# Note: The planet is not included in the simulation to focus on the binary star system dynamics. You can uncomment the planet and add it to the bodies list if you want to include it in the simulation.

# --- Simulation parameters and HDF5 writer setup ---
metadata = {
    "main": "binary_star_system.py",
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
    filename="results/simulation_testing_6.h5",         # Output file path
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
# Note: The binary star system will exhibit complex orbital dynamics due to the gravitational interaction between the two stars. You can analyze the resulting HDF5 file to study the trajectories and interactions of the stars over time.
# you can use analysis.py to visualize the results of the simulation.

# ================================ End of Binary Star System Simulation ================================