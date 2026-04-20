import numpy as np
from neutral_physics_engine.body import Body
from neutral_physics_engine.simulation import Simulation
from neutral_physics_engine.integrators import velocity_verlet
from neutral_physics_engine.gravity_field import GravityField
from neutral_physics_engine.io import HDF5Writer

# ================================ Random N-Body Simulation ================================
np.random.seed(42)

n_bodies = 10
bodies = []
# ------- Generate random bodies with more realistic parameters -------
for i in range(n_bodies):
    pos = np.random.uniform(-5e11, 5e11, size=3)          # ~±3 AU scale
    vel = np.random.uniform(-4e4, 4e4, size=3)            # reasonable orbital speeds
    mass = np.random.uniform(1e24, 2e30)                  # Earth to Solar mass range
    radius = np.random.uniform(1e6, 7e8)                  # Earth to Sun size
    body = Body(pos=pos, vel=vel, mass=mass, radius=radius)
    bodies.append(body)

# ------- Simulation parameters -------
T = 365.25 * 24 * 3600      # 1 year
dt = 3600 * 6               # start with larger initial dt (6 hours)

# ------- HDF5 writer setup and simulation execution -------
metadata = {
    "main": "random_n_bodies.py",
    "integrator": "velocity_verlet",
    "adaptive_step": True,
    "units_length": "meter",
    "units_mass": "kilogram",
    "units_time": "second"
}

with HDF5Writer(
    filename="results/random_n_bodies.h5",
    n_bodies=n_bodies,
    buffer_size=1000,
    compression="gzip",
    compression_opts=4,
    metadata=metadata
) as writer:

    gravity = GravityField(theta=0.7)        # higher theta = faster, still good enough
    sim = Simulation(bodies=bodies, field=gravity, integrator=velocity_verlet, dt=dt, hdf5_writer=writer)
    sim.run(T)

# The simulation will run for one year (T) with an initial time step of six hours (dt). The positions, velocities, and other relevant data of the bodies will be stored in the specified HDF5 file for later analysis and visualization.
# Note: The random N-body system will exhibit complex and chaotic dynamics due to the gravitational interactions between the bodies. You can analyze the resulting HDF5 file to study the trajectories and interactions of the bodies over time.
# you can use analysis.py to visualize the results of the simulation.

# ================================ End of Random N-Body Simulation ================================