"""
Theta Tradeoff Benchmark

Evaluates the accuracy vs. performance tradeoff of the Barnes-Hut algorithm
by varying the multiplexing parameter (theta). It compares Barnes-Hut O(N log N)
approximations against an exact O(N^2) direct force calculation baseline.
"""

import json
import numpy as np
from pathlib import Path
from time import perf_counter

from neutral_physics_engine.octree import Octree

# Physics constants and setup
G = 6.67430e-11  # Gravitational constant in m³ kg⁻¹ s⁻² (CODATA 2018 value)
EPS = 1e-10  # Small value used to avoid division-by-zero in distance calculations
np.random.seed(42)
N = 3000  # Fixed number of bodies for this benchmark
masses = np.random.uniform(1e20, 1e30, size=N)
positions = np.random.uniform(-1e11, 1e11, size=(N, 3))

thetas = [0.2, 0.5, 0.8, 1.0]
results = []

print("=" * 80)
print(f" THETA TRADEOFF BENCHMARK (N = {N})")
print(" Compares Barnes-Hut approximation errors/speeds against exact O(N^2) forces.")
print("=" * 80)
print("Calculating O(N^2) baseline (this may take a moment)...")

# --- Measure Exact Direct Calculation (Baseline) ---
direct_acc = np.zeros((N, 3))

start = perf_counter()

for i in range(N):
    for j in range(N):
        if i == j:
            continue

        # Direction vector and distance metrics
        dx = positions[j, 0] - positions[i, 0]
        dy = positions[j, 1] - positions[i, 1]
        dz = positions[j, 2] - positions[i, 2]
        dist_sq = dx * dx + dy * dy + dz * dz
        r = max(np.sqrt(dist_sq), EPS)
        inv_r = 1.0 / r
        inv_r3 = inv_r * inv_r * inv_r
        factor = G * masses[j] * inv_r3

        # Accumulate exact acceleration
        direct_acc[i, 0] += factor * dx
        direct_acc[i, 1] += factor * dy
        direct_acc[i, 2] += factor * dz

direct_time = perf_counter() - start

print(f"Baseline established in {direct_time:.4f}s.\n")

print(
    f"{'Theta':<8} | {'BH Time (s)':<15} | {'Direct (s)':<12} | {'Speedup':<10} | {'Relative Error'}"
)
print("-" * 80)

# --- Measure Barnes-Hut for Various Thetas ---
for theta in thetas:
    # Build the spatial partitioning tree
    tree = Octree(masses=masses, pos=positions, theta=theta)
    tree.build()

    bh_acc = np.zeros((N, 3))

    start = perf_counter()

    # Compute approximated accelerations
    for i in range(N):
        bh_acc[i] = tree.compute_acceleration(i)

    bh_time = perf_counter() - start

    # Calculate L2 norm relative error and performance gains
    error = np.linalg.norm(bh_acc - direct_acc) / np.linalg.norm(direct_acc)
    speedup = direct_time / bh_time

    results.append(
        {
            "theta": theta,
            "N": N,
            "bh_time": float(bh_time),
            "direct_time": float(direct_time),
            "speedup": float(speedup),
            "error": float(error),
        }
    )

    print(
        f"{theta:<8.2f} | {bh_time:<14.4f}s | {direct_time:<11.4f}s | {speedup:<9.2f}x | {error:.4e}"
    )

# --- Save Results ---
Path("benchmarks/results").mkdir(parents=True, exist_ok=True)

with open("benchmarks/results/theta_tradeoff.json", "w") as f:
    json.dump(results, f, indent=4)

print("=" * 80)
print("Results successfully saved to 'benchmarks/results/theta_tradeoff.json'\n")
