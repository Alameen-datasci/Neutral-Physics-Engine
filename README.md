# Neutral Physics Engine

A modular, high-precision physics engine written in Python for simulating gravitational N-body systems. This project implements Newtonian mechanics with support for multiple numerical integration schemes, collision resolution, and real-time energy analysis.

The current implementation demonstrates a **N-body simulation**, tracking trajectory dynamics and energy conservation throughout the event.

## 🚀 Key Features

* **Modular Numerical Integrators**: Support for swappable integration algorithms to balance speed vs. accuracy:
    * **Runge-Kutta 4 (RK4)**: High-order accuracy for complex orbital mechanics.
    * **Velocity Verlet**: Symplectic integrator offering superior energy conservation for long-duration simulations.
    * **Euler**: First-order baseline for performance benchmarking.
* **Physics Engine**:
    * Newtonian gravitational force calculation ($O(N^2)$).
    * Inelastic collision resolution with coefficient of restitution ($e=0.8$).
    * Positional correction (penetration resolution) to prevent numerical instability during impact.
* **Analytics & Visualization**:
    * Real-time tracking of Kinetic ($T$), Potential ($U$), and Total ($E$) energy to validate physical conservation laws.
    * Phase-space visualization (Speed vs. Time).
    * Matplotlib integration for trajectory plotting.

## 🛠️ Installation

This project requires **Python 3.8+** and the following scientific computing libraries.

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/Alameen-datasci/Neutral-Physics-Engine.git](https://github.com/Alameen-datasci/Neutral-Physics-Engine.git)
    cd nbody-simulation
    ```

2.  **Install dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r requirements.txt
    ```

## 💻 Usage

The entry point for the simulation is `main.py`. By default, it simulates a meteor drop scenario towards Earth.

To run the simulation:

```bash
python main.py