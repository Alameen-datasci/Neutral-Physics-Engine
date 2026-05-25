"""
Collision Stress Benchmark

This script isolates and evaluates the performance of the O(N log N) pair-finding
algorithm in the collision system. It scales the number of bodies (N) and measures
the time taken to identify potential collision pairs without advancing the physics state.
"""

import json
import numpy as np
from pathlib import Path
from time import perf_counter

from neutral_physics_engine.collision import CollisionSystem

# Set a fixed seed for reproducible random generation
np.random.seed(42)

Ns = [100, 500, 1000, 2000, 4000]
results = []

print("=" * 60)
print(" COLLISION STRESS BENCHMARK ")
print(" Isolating O(N log N) pair-finding performance.")
print("=" * 60)
print(f"{'Bodies (N)':<15} | {'Pairs Found':<15} | {'Time (s)'}")
print("-" * 60)


for N in Ns:
    # Initialize dummy properties for N bodies
    masses = np.ones(N)
    positions = np.random.uniform(-1000.0, 1000.0, size=(N, 3))
    velocities = np.zeros((N, 3))
    radii = np.ones(N) * 10.0

    # Instantiate the collision system with dummy data
    collision_system = CollisionSystem(
        masses=masses,
        positions=positions,
        velocities=velocities,
        radii=radii,
        restitution=0.8,
    )

    # Benchmark the pair-finding phase
    start = perf_counter()
    pairs = collision_system._get_collision_pairs()
    run_time = perf_counter() - start

    # Store metrics
    results.append(
        {"N": N, "pairs_found": len(pairs), "run_time_seconds": float(run_time)}
    )

    print(f"Bodies: {N:<5} | Pairs Found: {len(pairs):<6} | Time: {run_time:.4f}s")

# --- Save Results ---
output_dir = Path("benchmarks/results")
output_dir.mkdir(parents=True, exist_ok=True)

with open("benchmarks/results/collision_stress.json", "w") as f:
    json.dump(results, f, indent=4)

print("=" * 60)
print("Results successfully saved to 'benchmarks/results/collision_stress.json'\n")
