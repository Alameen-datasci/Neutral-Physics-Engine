"""
Barnes-Hut Scaling Benchmark (Python vs. C++)

Measures the algorithm complexity and time scaling as the number of bodies (N)
increases. It compares the individual components of the Barnes-Hut method
(Tree Building + Traversal) across both the legacy Python implementation and 
the optimized C++ implementation (via pybind11), validating them against the 
O(N^2) direct force computation.
"""

import json
import numpy as np
from pathlib import Path
from time import perf_counter

from neutral_physics_engine.octree import Octree
from neutral_physics_engine.octree_legacy import OctreeLegacy

# Physics constants
G = 6.67430e-11  # Gravitational constant in m³ kg⁻¹ s⁻² (CODATA 2018 value)
EPS = 1e-10  # Small value used to avoid division-by-zero in distance calculations

np.random.seed(42)
Ns = [100, 300, 1000, 3000]
results = []

# --- Benchmark Header Printout ---
print("=" * 120)
print(" BARNES-HUT SCALING BENCHMARK (PYTHON vs C++) ")
print(
    " Evaluating Octree Build and BH Traversal (Python & C++) vs. Direct O(N^2) N-body computation."
)
print("=" * 120)
print(
    f"{'N':<6} | {'Build (Py / C++)':<20} | {'Trav (Py / C++)':<20} | {'Direct O(N^2)':<14} | {'Speedup (Py/C++)':<18} | {'Error (C++)'}"
)
print("-" * 120)

def get_data():
    """Generates random masses and 3D positions for N bodies."""
    masses = np.random.uniform(1e20, 1e30, size=N)  # list of masses
    positions = np.random.uniform(-1e11, 1e11, size=(N, 3))  # Nx3 position matrix
    return masses, positions

for N in Ns:

    # --------------- Phase 1: Legacy Octree (Python) ---------------
    # Measure Tree Build Time and Traversal Time for the pure Python implementation
    masses, positions = get_data()
    tree_py = OctreeLegacy(masses=masses, pos=positions, theta=0.5)

    start = perf_counter()
    tree_py.build()
    build_time_py = perf_counter() - start

    bh_acc_py = np.zeros((N, 3))
    start = perf_counter()
    for i in range(N):
        bh_acc_py[i] = tree_py.compute_acceleration(i)
    bh_time_py = perf_counter() - start

    # --------------- Phase 2: Optimized Octree (C++) ---------------
    # Measure Tree Build Time and Traversal Time for the pybind11 C++ implementation
    masses, positions = get_data()
    tree_cpp = Octree(masses=masses, pos=positions, theta=0.5) 

    start = perf_counter()
    tree_cpp.build()
    build_time_cpp = perf_counter() - start

    bh_acc_cpp = np.zeros((N, 3))
    start = perf_counter()
    for i in range(N):
        bh_acc_cpp[i] = tree_cpp.compute_acceleration(i)
    bh_time_cpp = perf_counter() - start

    # --------------- Phase 3: Exact Direct Method O(N^2) ---------------
    # Compute exact forces directly in Python to establish baseline time and correctness
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

    # --------------- Phase 4: Compute Metrics & Log ---------------
    # Calculate relative errors and speedups against the O(N^2) baseline
    error_py = np.linalg.norm(bh_acc_py - direct_acc) / np.linalg.norm(direct_acc)
    error_cpp = np.linalg.norm(bh_acc_cpp - direct_acc) / np.linalg.norm(direct_acc)
    
    speedup_py = direct_time / bh_time_py
    speedup_cpp = direct_time / bh_time_cpp  # Fixed missing _cpp suffix here

    results.append(
        {
            "N": N,
            "build_time_python": float(build_time_py),
            "build_time_cpp": float(build_time_cpp),
            "bh_time_python": float(bh_time_py),
            "bh_time_cpp": float(bh_time_cpp),
            "direct_time": float(direct_time),
            "speedup_python": float(speedup_py),
            "speedup_cpp": float(speedup_cpp),
            "error_python": float(error_py),
            "error_cpp": float(error_cpp),
        }
    )

    # Print a formatted row comparing Py vs C++ metrics side-by-side
    print(
        f"{N:<6} | {build_time_py:>7.4f}s / {build_time_cpp:<8.4f}s | "
        f"{bh_time_py:>7.4f}s / {bh_time_cpp:<8.4f}s | "
        f"{direct_time:<14.4f} | "
        f"{speedup_py:>5.1f}x / {speedup_cpp:<6.1f}x | {error_cpp:.4e}"
    )

# --- Save Results ---
Path("benchmarks/results").mkdir(parents=True, exist_ok=True)

with open("benchmarks/results/scaling.json", "w") as f:
    json.dump(results, f, indent=4)

print("=" * 120)
print("Results successfully saved to 'benchmarks/results/scaling.json'\n")