# Neutral Physics Engine

[![Python Version](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/Alameen-datasci/Neutral-Physics-Engine.svg)](https://github.com/Alameen-datasci/Neutral-Physics-Engine/issues)
[![GitHub Stars](https://img.shields.io/github/stars/Alameen-datasci/Neutral-Physics-Engine.svg)](https://github.com/Alameen-datasci/Neutral-Physics-Engine/stargazers)

**Neutral Physics Engine** is a modular, high-performance Python library for simulating gravitational N-body systems using Newtonian mechanics. It combines classical integrators with a Barnes-Hut octree for efficient force calculations (O(N log N)), adaptive time-stepping, collision handling, and professional-grade HDF5 telemetry output.

Designed for **research, education, and professional use**, the engine is particularly suited for studying planetary dynamics, binary star systems, star clusters, and other gravitational problems where long-term stability and energy/momentum conservation are critical.

## 🚀 Key Features

- **Efficient Gravitational Solver**: Barnes-Hut octree implementation enabling scalable N-body simulations (far beyond O(N²) limits).
- **Numerical Integrators**:
  - Euler (first-order)
  - Runge-Kutta 4 (high accuracy)
  - **Velocity Verlet** (symplectic – excellent long-term energy conservation)
- **Adaptive Time-Stepping**: Automatic step-size control based on local truncation error for handling close encounters and stiff dynamics.
- **Collision Detection & Resolution**: Octree-accelerated sphere collision detection with impulse-based response and positional correction.
- **High-Performance I/O**: Buffered, compressed HDF5 logging of positions, velocities, energies, and momenta.
- **Comprehensive Analysis Tools**: Energy conservation diagnostics, relative error plots, momentum tracking, and 2D/3D trajectory projections.
- **Modular & Extensible**: Clean object-oriented design with clear separation of concerns (Body, GravityField, Simulation, Analysis, etc.).

## 🛠️ Installation

### 1. Clone the Repository
```bash
git clone https://github.com/Alameen-datasci/Neutral-Physics-Engine.git
cd Neutral-Physics-Engine
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Install the Package (Recommended)
```bash
pip install -e .
```

This allows you to import the engine as neutral_physics_engine from anywhere.

## 💻 Quick Start
### Example: Inner Solar System (1-year simulation)
```python
from neutral_physics_engine.body import Body
from neutral_physics_engine.simulation import Simulation
from neutral_physics_engine.gravity_field import GravityField
from neutral_physics_engine.integrators import velocity_verlet
from neutral_physics_engine.io import HDF5Writer

# Define bodies
sun = Body(mass=1.989e30, pos=[0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0], radius=6.9634e8)
mercury = Body(mass=3.301e23, pos=[5.79e10, 0.0, 0.0], vel=[0.0, 47400.0, 0.0], radius=2.44e6)
# ... add Venus, Earth, Mars similarly

bodies = [sun, mercury, venus, earth, mars]

with HDF5Writer(
    filename="results/inner_solar_system.h5",
    n_bodies=len(bodies),
    buffer_size=1000,
    metadata={"simulation": "inner_solar_system", "integrator": "velocity_verlet"}
) as writer:

    gravity = GravityField(theta=0.5)
    sim = Simulation(
        bodies=bodies,
        field=gravity,
        integrator=velocity_verlet,
        dt=3600,                    # 1 hour initial step
        hdf5_writer=writer
    )
    sim.run(365.25 * 24 * 3600)     # 1 year
```

### Analysis
```python
from neutral_physics_engine.analysis import Analysis

with Analysis("results/inner_solar_system.h5") as analysis:
    analysis.relative_energy_error()
    analysis.plot_energy_components()
    analysis.plot_projection(planes=["xy", "xz"])
    print("Energy drift rate:", analysis.energy_drift_rate())
```

## 📊 Scientific Validation
The engine conserves linear and angular momentum to machine precision and exhibits bounded energy error when using the symplectic Velocity Verlet integrator. Barnes-Hut acceleration with `theta ≈ 0.5` provides an excellent accuracy–performance trade-off for large-N systems.
## 📚 Academic & Research Use Cases

- Celestial mechanics and orbital dynamics
- Star cluster evolution
- Planetary formation studies
- Spacecraft trajectory planning
- Educational demonstrations of numerical methods and conservation laws

See `examples/` folder for ready-to-run simulations (Sun-Earth, Inner Solar System, Binary Stars, Random N-body).

## 🔬 Skills Demonstrated

- Advanced numerical methods in computational physics
- High-performance spatial algorithms (Barnes-Hut octree)
- Symplectic integration and adaptive stepping
- Scientific data management (HDF5)
- Modular software architecture for research tools

---
See `CHANGELOG.md` for the complete development history.
---