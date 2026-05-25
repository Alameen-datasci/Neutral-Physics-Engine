"""
HDF5 I/O Performance Benchmark

Measures the file write performance and file size footprint of the simulation
engine when exporting telemetry to HDF5. Varies the in-memory buffer size to
observe flush-to-disk overhead tradeoffs.
"""

import json
import numpy as np
from pathlib import Path
from time import perf_counter

from neutral_physics_engine.gravity_field import GravityField
from neutral_physics_engine.simulation import Simulation
from neutral_physics_engine.integrators import euler
from neutral_physics_engine.io import HDF5Writer
from neutral_physics_engine.body import Body

# Setup consistent random starting state
np.random.seed(42)
N = 200
masses = np.random.uniform(1e20, 1e24, size=N)
positions = np.random.uniform(-1e11, 1e11, size=(N, 3))
velocities = np.random.uniform(-1e4, 1e4, size=(N, 3))
radii = 1000.0


def get_fresh_bodies():
    """Returns a fresh array of body instances for a clean simulation start."""
    return [
        Body(mass=masses[i], pos=positions[i], vel=velocities[i], radius=radii)
        for i in range(N)
    ]


dt = 3600.0
T = 200 * dt
results = []
field = GravityField(theta=0.5)

# Array of chunk sizes to test
buffer_sizes = [1, 10, 100, 200]

# Create a directory to hold the temporary HDF5 benchmark files
output_dir = Path("benchmarks/results/io_test")
output_dir.mkdir(parents=True, exist_ok=True)

print("=" * 65)
print(" HDF5 I/O BUFFER PERFORMANCE BENCHMARK ")
print(f" Bodies: {N} | Total Steps: {int(T / dt)} | Integrator: Euler")
print("=" * 65)
print(f"{'Buffer Size':<15} | {'Write Time (s)':<18} | {'File Size (MB)'}")
print("-" * 65)

for buffer in buffer_sizes:
    bodies = get_fresh_bodies()

    h5_path = output_dir / f"test_buf_{buffer}.h5"

    # Clean up previous runs if they exist
    if h5_path.exists():
        h5_path.unlink()

    writer = HDF5Writer(filename=h5_path, n_bodies=N, buffer_size=buffer)

    # Benchmark the simulation loop strictly regarding I/O overhead
    with writer:
        sim = Simulation(
            bodies=bodies,
            integrator=euler,
            field=field,
            dt=dt,
            time_stepping="fixed",
            enable_collisions=False,
            hdf5_writer=writer,
        )

        start = perf_counter()
        sim.run(T)
        run_time = perf_counter() - start

    # Measure footprint
    file_size_mb = h5_path.stat().st_size / (1024 * 1024)

    results.append(
        {
            "buffer_size": buffer,
            "run_time_seconds": float(run_time),
            "file_size_mb": float(file_size_mb),
        }
    )

    print(f"{buffer:<15} | {run_time:<17.4f}s | {file_size_mb:.2f} MB")

# --- Save Results ---
Path("benchmarks/results").mkdir(parents=True, exist_ok=True)

with open("benchmarks/results/hdf5_performance.json", "w") as f:
    json.dump(results, f, indent=4)

print("=" * 65)
print("Results successfully saved to 'benchmarks/results/hdf5_performance.json'\n")
