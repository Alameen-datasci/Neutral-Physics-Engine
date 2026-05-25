"""
Integrator Comparison Benchmark

Analyzes the mathematical stability and runtime of various numerical integrators
(Euler, Runge-Kutta 4, Velocity Verlet). Evaluates energy conservation (drift)
using a classic Earth-Sun orbit over 365 steps.
"""

import json
import numpy as np
from pathlib import Path
from time import perf_counter

from neutral_physics_engine.gravity_field import GravityField
from neutral_physics_engine.integrators import euler, rk4, velocity_verlet

N = 2
# Sun-Earth configurations for a visible output/benchmark
masses = np.array([1.989e30, 5.972e24], dtype=np.float64)
positions = np.array([[0.0, 0.0, 0.0], [1.496e11, 0.0, 0.0]], dtype=np.float64)
velocities = np.array([[0.0, 0.0, 0.0], [0.0, 29780.0, 0.0]], dtype=np.float64)

# Pack state matrix into 1D vector form expected by solvers
base_state = np.concatenate((positions.flatten(), velocities.flatten()))
field = GravityField(theta=0.5)


def compute_total_energy(masses, state_vector, n_bodies):
    """Calculates the combined kinetic and potential energy of the current state."""
    pos = state_vector[: n_bodies * 3].reshape(n_bodies, 3)
    vel = state_vector[n_bodies * 3 :].reshape(n_bodies, 3)
    ke = 0.5 * np.sum(masses * np.sum(vel**2, axis=1))
    pe = field.compute_potential(masses, pos)
    return ke + pe


# Timestepping variables (1 step/day for a year)
dt = 86400.0
steps = 365

integrators = [
    {"name": "euler", "func": euler},
    {"name": "rk4", "func": rk4},
    {"name": "velocity_verlet", "func": velocity_verlet},
]

results = []

print("=" * 65)
print(" NUMERICAL INTEGRATOR BENCHMARK ")
print(f" System: Sun-Earth | Steps: {steps} | dt: 1 day")
print("=" * 65)
print(f"{'Integrator':<20} | {'Time (s)':<15} | {'Energy Drift'}")
print("-" * 65)

for integ in integrators:
    state = base_state.copy()
    initial_energy = compute_total_energy(masses, state, N)

    t = 0.0
    start = perf_counter()

    # Step forward in time
    for _ in range(steps):
        state = integ["func"](masses, state, field, dt, t)
        t += dt

    run_time = perf_counter() - start

    final_energy = compute_total_energy(masses, state, N)

    # Relative energy drift calculation
    energy_drift = abs(final_energy - initial_energy) / abs(initial_energy)

    results.append(
        {
            "integrator": integ["name"],
            "N": N,
            "dt": float(dt),
            "steps": steps,
            "run_time": float(run_time),
            "energy_drift": float(energy_drift),
        }
    )

    print(f"{integ['name']:<20} | {run_time:<14.4f}s | {energy_drift:.4e}")

# --- Save Results ---
Path("benchmarks/results/").mkdir(parents=True, exist_ok=True)

with open("benchmarks/results/integrator_comparison.json", "w") as f:
    json.dump(results, f, indent=4)

print("=" * 65)
print("Results successfully saved to 'benchmarks/results/integrator_comparison.json'\n")
