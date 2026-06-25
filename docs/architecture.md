# Architecture

## Overview

Neutral Physics Engine is organized around a central `Simulation` class that coordinates body management, numerical integration, gravitational field evaluation, collision handling, and telemetry output. The architecture separates physical state, numerical methods, spatial acceleration structures, and storage systems into independent modules, allowing individual subsystems to evolve without affecting the entire codebase.

At a high level, users define physical bodies, configure a simulation, select a numerical integrator, and execute the simulation for a desired duration. During execution, gravitational accelerations are computed through a Barnes-Hut octree backend implemented in C++, while simulation data can optionally be streamed to compressed HDF5 files for later analysis.

---

## Core Components

### Body (`body.py`)

The `Body` class represents a physical object within the simulation. Examples include planets, stars, asteroids, meteors, or arbitrary point masses.

Each body stores:

* Mass
* Position vector
* Velocity vector
* Radius

Bodies themselves contain no simulation logic. They serve only as containers for physical state.

---

### Simulation (`simulation.py`)

The `Simulation` class is the central orchestrator of the engine.

Its responsibilities include:

* Managing all bodies
* Advancing the simulation through time
* Performing adaptive or fixed time stepping
* Invoking numerical integrators
* Computing energy and momentum diagnostics
* Handling collisions
* Coordinating telemetry output

Users provide a list of `Body` objects, a gravitational field implementation, and an integration method. The simulation then repeatedly advances the system until the requested end time is reached.

---

### Integrators (`integrators.py`)

Numerical integrators are responsible for advancing the state of the system through time.

The engine currently provides:

* Euler
* Runge-Kutta 4 (RK4)
* Velocity Verlet

Integrators operate on a flattened state vector rather than directly modifying body objects. Given the current state and a timestep, an integrator computes a new state vector representing updated positions and velocities.

The integrator itself does not calculate gravitational forces. Instead, it requests accelerations from the gravitational field whenever required.

---

### Gravity Field (`gravity_field.py`)

The `GravityField` class provides gravitational accelerations and potential energy evaluations.

It acts as an abstraction layer between numerical integrators and the underlying spatial acceleration structure.

Responsibilities include:

* Computing gravitational accelerations
* Computing approximate gravitational potential energy
* Managing octree construction and reuse during a timestep

From the perspective of the integrator, the gravitational field behaves like a callable function that returns accelerations for a given particle configuration.

---

### Barnes-Hut Octree (`octree.cpp`)

The computational core of the engine is the Barnes-Hut octree implemented in C++17.

The octree is responsible for:

* Spatial partitioning
* Center-of-mass aggregation
* Gravitational force approximation
* Broad-phase collision detection

Instead of evaluating every pairwise interaction directly, the Barnes-Hut algorithm approximates distant particle groups as single aggregated masses. This reduces computational complexity from O(N²) to approximately O(N log N).

The octree is exposed to Python through Pybind11 and is used transparently by the gravity and collision systems.

---

### Collision System (`collision.py`)

The collision subsystem performs two-stage collision detection:

1. Broad-phase filtering using the octree
2. Narrow-phase sphere intersection tests

Once a collision is detected, impulse-based collision resolution updates the velocities of the affected bodies while applying positional correction to prevent overlap.

Collision handling can be enabled or disabled depending on the simulation scenario.

---

### HDF5 Writer (`io.py`)

The `HDF5Writer` provides high-performance telemetry storage.

Instead of writing simulation data to disk every timestep, data is first accumulated in memory buffers. Once a configurable buffer limit is reached, the data is written to compressed and chunked HDF5 datasets.

Stored telemetry includes:

* Positions
* Velocities
* Energy histories
* Momentum histories
* Simulation metadata

This design significantly reduces disk I/O overhead during long simulations.

---

### Analysis (`analysis.py`)

The analysis module operates independently from the simulation engine.

Its responsibilities include:

* Reading HDF5 telemetry
* Plotting trajectories
* Computing energy drift
* Visualizing momentum conservation
* Generating diagnostic figures

Separating analysis from simulation ensures that visualization overhead does not impact runtime performance.

---

## Simulation Pipeline

The complete execution flow is illustrated below:

```text
User Creates Bodies
        │
        ▼
Simulation Initialization
        │
        ▼
State Vector Construction
        │
        ▼
Integrator Step
        │
        ▼
GravityField Evaluation
        │
        ▼
C++ Barnes-Hut Octree
        │
        ▼
Acceleration Computation
        │
        ▼
New State Vector
        │
        ▼
Update Body States
        │
        ▼
Collision Handling
        │
        ▼
Diagnostics Computation
        │
        ▼
Optional HDF5 Logging
```

---

## State Vector Representation

To simplify integration and improve compatibility with numerical solvers, the simulation internally stores the system state as a flattened vector.

The vector consists of:

```text
[pos_x, pos_y, pos_z] × N
[vel_x, vel_y, vel_z] × N
```

where N is the number of bodies in the simulation.

Integrators operate exclusively on this representation and return an updated state vector after each timestep.

---

## Python–C++ Interaction

The engine uses a hybrid architecture.

Python is responsible for:

* User API
* Simulation orchestration
* Numerical workflow management
* Data analysis
* Telemetry management

C++ is responsible for:

* Octree construction
* Barnes-Hut traversal
* Center-of-mass aggregation
* Force evaluation
* Broad-phase collision queries

Pybind11 provides the bridge between both layers, allowing Python code to invoke native C++ routines while preserving a simple user-facing API.

Recent versions of the engine further reduced Python/C++ communication overhead through a zero-copy memory interface that accesses NumPy memory directly from C++.

---

## Design Philosophy

The architecture was designed around four primary goals:

1. Separation of concerns between physics, numerics, storage, and visualization.
2. Scalability through Barnes-Hut spatial acceleration.
3. Performance through native C++ execution of computational hot paths.
4. Extensibility for future work in scientific computing, high-performance simulation, and advanced physical modeling.

This modular structure allowed the project to evolve from a simple particle simulator into a computational physics platform supporting adaptive integration, spatial acceleration, collision systems, telemetry pipelines, and native C++ performance optimization.
