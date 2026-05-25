from time import perf_counter

from neutral_physics_engine.body import Body
from neutral_physics_engine.simulation import Simulation
from neutral_physics_engine.integrators import velocity_verlet
from neutral_physics_engine.gravity_field import GravityField
from neutral_physics_engine.io import HDF5Writer

# ------------------------------------- Create bodies --------------------------------------
sun = Body(mass=1.989e30, pos=[0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0], radius=6.96e8)
earth = Body(
    mass=5.972e24, pos=[1.49e11, 0.0, 0.0], vel=[0.0, 29780.0, 0.0], radius=6.371e6
)
bodies = [sun, earth]

# ------------------------------------- Run simulation --------------------------------------
T = 365.25 * 24 * 3600  # Total simulation time (1 year in seconds)
dt = 60 * 60  # Time step (1 hour in seconds)

metadata = {
    "main": "sun_earth.py",
    "integrator": "velocity_verlet",
    "time_stepping": "adaptive",
    "units_length": "meter",
    "units_mass": "kilogram",
    "units_time": "second",
}

print("\n" + "=" * 65)
print(" 🌍 NEUTRAL PHYSICS ENGINE: Sun-Earth System")
print("=" * 65)
print(f" [*] Total Bodies : {len(bodies)}")
print(f" [*] Sim Duration : {T} seconds (~{T/(24*3600):.1f} days)")
print(f" [*] Initial dt   : {dt} seconds")
print(f" [*] Output File  : results/sun_earth.h5")
print("-" * 65)
print(" [>] Initializing simulation environment...")

with HDF5Writer(
    filename="results/sun_earth.h5",
    n_bodies=len(bodies),
    buffer_size=1000,
    compression="gzip",
    compression_opts=4,
    metadata=metadata,
) as writer:

    gravity = GravityField(theta=0.5)
    sim = Simulation(
        bodies=bodies,
        field=gravity,
        integrator=velocity_verlet,
        dt=dt,
        time_stepping="adaptive",
        hdf5_writer=writer,
        enable_collisions=False,
    )

    print(" [>] Running simulation... Please wait.")
    start = perf_counter()

    sim.run(T)

    run_time = perf_counter() - start

print("-" * 65)
print(f" [✓] Simulation completed successfully in {run_time:.2f} seconds!")
print("=" * 65 + "\n")
