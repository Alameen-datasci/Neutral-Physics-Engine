"""
Adaptive Stepping Benchmark

This script compares the performance and accuracy (energy drift) of adaptive
time-stepping versus fixed time-stepping using the RK4 integrator. The test
case is a high-eccentricity Sun-Comet orbital system simulated over 1 year.
"""

import json
import numpy as np
from pathlib import Path
from time import perf_counter

from neutral_physics_engine.body import Body
from neutral_physics_engine.integrators import rk4
from neutral_physics_engine.simulation import Simulation
from neutral_physics_engine.gravity_field import GravityField

# Define celestial bodies
sun = Body(mass=1.989e30, pos=[0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0], radius=6.96e8)

comet = Body(mass=1e14, pos=[1.496e11, 0.0, 0.0], vel=[0.0, 5000.0, 0.0], radius=1000.0)


def get_fresh_bodies():
    """
    Returns a fresh instantiation of the celestial bodies to ensure
    independent states between simulation runs.
    """
    return [
        Body(mass=sun.mass, pos=sun.pos, vel=sun.vel, radius=sun.radius),
        Body(mass=comet.mass, pos=comet.pos, vel=comet.vel, radius=comet.radius),
    ]


# Simulation parameters (1 Year total, 10-day initial step)
dt = 86400.0 * 10
T = 86400.0 * 365
results = []
field = GravityField(theta=0.5)

print("=" * 65)
print(" ADAPTIVE VS FIXED STEPPING BENCHMARK ")
print(" Simulation Time: 1 Year | Initial dt: 10 Days | System: Sun-Comet")
print("=" * 65)
print(f"{'Mode':<15} | {'Run Time (s)':<15} | {'Energy Drift'}")
print("-" * 65)

# --- 1. Adaptive Stepping Benchmark ---
bodies_adaptive = get_fresh_bodies()
sim_adaptive = Simulation(
    bodies=bodies_adaptive,
    integrator=rk4,
    field=field,
    dt=dt,
    time_stepping="adaptive",
    enable_collisions=False,
)
initial_energy_ad = sim_adaptive.total

start = perf_counter()

sim_adaptive.run(T)

adaptive_time = perf_counter() - start

final_energy_ad = sim_adaptive.total
error_adaptive = abs(final_energy_ad - initial_energy_ad) / abs(initial_energy_ad)

results.append(
    {
        "mode": "adaptive",
        "initial_dt": dt,
        "run_time": adaptive_time,
        "energy_drift": error_adaptive,
    }
)

print(f"{'Adaptive':<15} | {adaptive_time:<14.4f}s | {error_adaptive:.4e}")

# --- 2. Fixed Stepping Benchmark ---
bodies_fixed = get_fresh_bodies()
sim_fixed = Simulation(
    bodies=bodies_fixed,
    integrator=rk4,
    field=field,
    dt=dt,
    time_stepping="fixed",
    enable_collisions=False,
)
initial_energy_fix = sim_fixed.total

start = perf_counter()

sim_fixed.run(T)

fixed_time = perf_counter() - start

final_energy_fix = sim_fixed.total
error_fixed = abs(final_energy_fix - initial_energy_fix) / abs(initial_energy_fix)

results.append(
    {
        "mode": "fixed",
        "initial_dt": dt,
        "run_time": fixed_time,
        "energy_drift": error_fixed,
    }
)

print(f"{'Fixed':<15} | {fixed_time:<14.4f}s | {error_fixed:.4e}")

# --- Save Results ---
Path("benchmarks/results").mkdir(parents=True, exist_ok=True)

with open("benchmarks/results/adaptive_stepping.json", "w") as f:
    json.dump(results, f, indent=4)

print("=" * 65)
print("Results successfully saved to 'benchmarks/results/adaptive_stepping.json'\n")
