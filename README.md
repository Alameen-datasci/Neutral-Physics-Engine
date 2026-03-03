# Neutral Physics Engine

[![Python Version](https://img.shields.io/badge/python-3.8%2B-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![GitHub Issues](https://img.shields.io/github/issues/Alameen-datasci/Neutral-Physics-Engine.svg)](https://github.com/Alameen-datasci/Neutral-Physics-Engine/issues)
[![GitHub Stars](https://img.shields.io/github/stars/Alameen-datasci/Neutral-Physics-Engine.svg)](https://github.com/Alameen-datasci/Neutral-Physics-Engine/stargazers)

A modular, high-precision Python physics engine for simulating gravitational N-body systems using Newtonian mechanics. It supports multiple numerical integrators, collision handling, adaptive time-stepping, and comprehensive post-simulation analysis for energy and momentum conservation. Ideal for educational purposes, research prototypes, or exploring celestial mechanics like planetary orbits.

The engine powers simulations such as Sun-Earth orbital dynamics, demonstrating long-term stability and physical accuracy.

## 🚀 Key Features

- **Flexible Numerical Integrators**:
  - **Euler**: Simple first-order method for quick prototyping.
  - **Runge-Kutta 4 (RK4)**: High-accuracy for complex trajectories.
  - **Velocity Verlet**: Symplectic integrator for excellent energy conservation in Hamiltonian systems.
- **Core Physics**:
  - Newtonian gravity with O(N²) force calculations.
  - Collision detection and resolution with configurable coefficient of restitution (default: 0.8).
  - Adaptive time-stepping for handling stiff dynamics (e.g., close encounters) while optimizing performance.
- **Advanced Analysis & Visualization**:
  - Real-time logging of kinetic, potential, and total energy.
  - Momentum tracking (linear and angular) relative to the center of mass.
  - Post-simulation tools: Relative energy error, drift rate, 3D trajectory projections (xy, xz, yz), and momentum plots.
  - Built-in Matplotlib integration for insightful visualizations.
- **Extensibility**:
  - Easy to add custom bodies, forces, or integrators.
  - Placeholders for rotational dynamics (quaternions and angular velocity).
- **Performance & Stability**:
  - NumPy-optimized computations.
  - Error tolerances and safety factors for adaptive stepping.

## 🛠️ Installation

### Prerequisites
- Python 3.8 or higher.
- Virtual environment recommended (e.g., via `venv` or `conda`).

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/Alameen-datasci/Neutral-Physics-Engine.git
    cd Neutral-Physics-Engine
    ```

2. **Set Up a Virtual Environment** (optional but recommended):
    ```bash
    python -m venv .venv
    source .venv/bin/activate  # On Unix/macOS
    .venv\Scripts\activate     # On Windows
    ```

3. **Install Dependencies:**
The project uses NumPy for computations and Matplotlib for plotting. Install via:
    ```bash
    pip install -r requirements.txt
    ```
    (If requirements.txt is missing, run pip install numpy matplotlib.)

## 💻 Quick Start
Run the default simulation (a year-long Sun-Earth orbital model using RK4):
```bash
python main.py
```

This will:

- Simulate the system for ~31.5 million seconds (1 year).
- Generate analysis plots: Energy error, components, trajectory projections, and momentum conservation.
- Output the energy drift rate for validation.

### Customizing the Simulation
Edit `main.py` to experiment:

- **Switch Integrators:** Replace integrator=rk4 with euler or velocity_verlet.
- **Add Bodies:** Extend the bodies list with new Body instances.
- **Adjust Parameters:** Modify dt (initial time step), T (total time), or restitution.

Example snippet for a custom three-body system:
```python
from body import Body
from simulation import Simulation
from analysis import Analysis
from integrators import velocity_verlet
from forces import forces

# Define bodies (e.g., Sun, Earth, Moon)
sun = Body(mass=1.989e30, pos=[0, 0, 0], vel=[0, 0, 0], radius=6.96e8)
earth = Body(mass=5.972e24, pos=[1.49e11, 0, 0], vel=[0, 29780, 0], radius=6.371e6)
moon = Body(mass=7.342e22, pos=[1.49e11 + 3.84e8, 0, 0], vel=[0, 29780 + 1022, 0], radius=1.737e6)
bodies = [sun, earth, moon]

# Run simulation
sim = Simulation(bodies=bodies, integrator=velocity_verlet, force_fn=forces, dt=3600)
sim.run(365.25 * 24 * 3600)  # 1 year

# Analyze
analysis = Analysis(sim)
analysis.plot_projection(planes=["xy"])
analysis.relative_energy_error()
```

## 📊 Example Outputs

- **Trajectory Projections:** 2D views of orbits in specified planes.
- **Energy Plots:** Demonstrate conservation with minimal drift using advanced integrators.
- **Momentum Graphs:** Verify linear and angular momentum stability.

Running the simulation generates these visuals directly, showcasing the project's analytical capabilities.

## 🔍 Skills Demonstrated

- **Computational Physics:** Implementation of gravitational forces, integrators, and conservation laws.
- **Software Design:** Modular architecture with classes for bodies, simulations, and analysis.
- **Numerical Methods:** Adaptive time-stepping and error control for efficient, accurate simulations.
- **Data Visualization:** Using Matplotlib for professional-grade plots.
- **Problem-Solving:** Handling edge cases like collisions and singularities.

This project can be highlighted in resumes or portfolios to illustrate hands-on experience in building simulation tools.

See `CHANGELOG.md` for development history.