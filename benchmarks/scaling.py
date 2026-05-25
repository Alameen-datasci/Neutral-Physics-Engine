"""
Barnes-Hut Scaling Benchmark

Measures the algorithm complexity and time scaling as the number of bodies (N)
increases. It compares the individual components of the Barnes-Hut method
(Tree Building + Traversal) against the O(N^2) direct force computation.
"""

import json
import numpy as np
from pathlib import Path
from time import perf_counter

from neutral_physics_engine.octree import Octree

# Physics constants
G = 6.67430e-11  # Gravitational constant in m³ kg⁻¹ s⁻² (CODATA 2018 value)
EPS = 1e-10  # Small value used to avoid division-by-zero in distance calculations

np.random.seed(42)
Ns = [100, 300, 1000, 3000]
results = []

print("=" * 105)
print(" BARNES-HUT SCALING BENCHMARK ")
print(
    " Evaluating Octree Build, BH Traversal, and Direct N-body computation across increasing N."
)
print("=" * 105)
print(
    f"{'N':<6} | {'Tree Build':<12} | {'BH Travers':<12} | {'Direct O(N^2)':<14} | {'Speedup':<9} | {'Relative Error'}"
)
print("-" * 105)

for N in Ns:
    # Setup test vectors
    masses = np.random.uniform(1e20, 1e30, size=N)  # list of masses
    positions = np.random.uniform(-1e11, 1e11, size=(N, 3))  # Nx3 position matrix

    # Phase 1: Measure Tree Build Time
    tree = Octree(masses=masses, pos=positions, theta=0.5)  # we passed the values

    start = perf_counter()
    tree.build()
    build_time = perf_counter() - start

    # Phase 2: Measure Barnes-Hut Traversal Time
    bh_acc = np.zeros((N, 3))
    start = perf_counter()
    for i in range(N):
        bh_acc[i] = tree.compute_acceleration(i)
    bh_time = perf_counter() - start

    # Phase 3: Measure Exact Direct Time (O(N^2))
    direct_acc = np.zeros((N, 3))
    start = perf_counter()
    for i in range(N):
        for j in range(N):
            if i == j:
                continue

            dx = positions[j, 0] - positions[i, 0]
            dy = positions[j, 1] - positions[i, 1]
            dz = positions[j, 2] - positions[i, 2]
            dist_sq = dx * dx + dy * dy + dz * dz
            r = max(np.sqrt(dist_sq), EPS)
            inv_r = 1.0 / r
            inv_r3 = inv_r * inv_r * inv_r
            factor = G * masses[j] * inv_r3

            direct_acc[i, 0] += factor * dx
            direct_acc[i, 1] += factor * dy
            direct_acc[i, 2] += factor * dz

    direct_time = perf_counter() - start

    # Phase 4: Compute Metrics
    error = np.linalg.norm(bh_acc - direct_acc) / np.linalg.norm(direct_acc)
    speedup = direct_time / bh_time

    results.append(
        {
            "N": N,
            "build_time": float(build_time),
            "bh_time": float(bh_time),
            "direct_time": float(direct_time),
            "speedup": float(speedup),
            "error": float(error),
        }
    )

    print(
        f"{N:<6} | {build_time:<11.4f}s | {bh_time:<11.4f}s | {direct_time:<13.4f}s | {speedup:<8.2f}x | {error:.4e}"
    )

# --- Save Results ---
Path("benchmarks/results").mkdir(parents=True, exist_ok=True)

with open("benchmarks/results/scaling.json", "w") as f:
    json.dump(results, f, indent=4)

print("=" * 105)
print("Results successfully saved to 'benchmarks/results/scaling.json'\n")
