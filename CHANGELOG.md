# Changelog

All notable changes to the "Neutral Physics Engine" project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased] / [v4.0.0]
### Added
- **Barnes-Hut Octree (`octree.py`)**: Full O(N log N) spatial partitioning implementation with automatic tree construction, recursive insertion, center-of-mass aggregation, and max-radius tracking for collision culling.
- **GravityField (`gravity_field.py`)**: High-level callable interface for gravitational accelerations and total potential energy. Includes automatic octree caching to avoid redundant rebuilds within a single time step.
- **CollisionSystem (`collision.py`)**: New dedicated collision module using the octree for broad-phase detection (O(N log N)) followed by narrow-phase sphere intersection checks and impulse-based resolution with positional correction.
- **HDF5Writer (`io.py`)**: High-performance buffered, chunked, compressed HDF5 telemetry output with context-manager support, automatic flushing, metadata tagging, and extendable datasets for positions, velocities, energies, and momenta.
- **Full modular collision pipeline**: Collision detection and resolution now delegated from `Simulation` to `CollisionSystem` (uses octree + per-body radii).
- **Potential energy via Barnes-Hut**: `GravityField.compute_potential` provides fast approximate total gravitational potential (used by `_compute_energy`).
- **Context-managed Analysis**: `analysis.py` now opens HDF5 files as a context manager, reads pre-computed telemetry, and includes `get_metadata()` and `get_integrator_name()` helpers.

### Changed
- **Simulation class (`simulation.py`)**: 
  - `__init__` now accepts `field: GravityField` and optional `hdf5_writer: HDF5Writer`.
  - `_handle_collisions` completely refactored to use `CollisionSystem`.
  - Energy calculation now uses `field.compute_potential` (Barnes-Hut) instead of the old O(N²) loop.
  - `_log_state` now pushes data to HDF5Writer when provided.
  - Pre-computed `self.pos`/`self.vel` arrays updated after every integration and collision step.
- **Integrators (`integrators.py`)**: All integrators (`euler`, `rk4`, `velocity_verlet`) now accept a `field` callable instead of the old `force_fn`. Updated type hints and docstrings for clarity.
- **Analysis (`analysis.py`)**: 
  - Now reads directly from HDF5 datasets (`positions`, `velocities`, `energy/*`, `momentum/*`).
  - `plot_projection` fixed to load the full position history once from disk.
  - Added grid lines and improved layout for all plots.
- **Body (`body.py`)**: Minor type consistency updates and docstring polish (rotational placeholders remain commented).
- **Package structure**: All core code is now under the recommended `src/neutral_physics_engine/` layout (as planned in v3.0.1).
- **Adaptive stepping and logging**: Minor robustness improvements and updated comments throughout.

### Removed
- Old `forces.py` (O(N²) direct summation) — fully replaced by Barnes-Hut `Octree` + `GravityField`.
- Legacy collision logic inside `simulation.py` (the old `_handle_collisions` and `_resolve_collision` methods).
- Direct potential-energy O(N²) loop in `_compute_energy` (now delegated to `GravityField`).
- Old direct `force_fn` usage in integrators and simulation (replaced by `field` callable).

### Fixed
- Several small bugs in octree traversal, collision normal handling, and positional correction (added `slop` parameter and better singularity guards).
- `plot_projection` in Analysis now correctly handles large HDF5 datasets without loading everything into memory multiple times.
- Collision pair ordering (`i < j`) and velocity update signs fixed for physical correctness.

### Note
- The `tests/` folder containing the previous pytest files is **invalid** right now. All test files are out of date due to the major refactoring (new package layout, Barnes-Hut API, HDF5 I/O, CollisionSystem, GravityField, etc.). Tests will be updated in a future patch release.

---

## [v3.0.1] (previous release)
### Added
- **Modern Package Layout:** Adopted the recommended `src/` layout (`src/neutral_physics_engine/`) for better separation between library code and development/test files.
- **Initial Test Suite:** Added a `tests/` directory with basic unit tests:
  - `test_body.py`: validation of Body initialization and vector shapes
  - `test_forces.py`: symmetry and correctness of gravitational accelerations
  - `test_integrator.py`: basic smoke tests for integrator steps (state shape preservation, simple cases)
  - `test_simulation.py`: energy and momentum conservation checks in short runs
- **Testing Infrastructure:** Added placeholder `__init__.py` files and initial pytest-compatible structure.

### Changed
- **Project Structure Reorganization:** Moved all core package code into `src/neutral_physics_engine/`; updated import paths accordingly.
- **Documentation & Maintainability:** Minor updates to module docstrings and README references to reflect the new layout.

### Fixed
- No functional changes to simulation logic, physics, or public API — purely structural and testing improvements.

**No breaking changes** — this is a patch release (3.0.1).

---

## [v3.0]
### Added
- **Analysis Module:** Introduced a new `analysis.py` module with an `Analysis` class for post-simulation evaluation, including methods for calculating and plotting relative energy error, energy components (kinetic, potential, total), energy drift rate, trajectory projections (xy, xz, yz planes), and magnitudes of linear and angular momentum over time.
- **Adaptive Time-Stepping:** Implemented an adaptive time-step mechanism in `simulation.py` using error estimation (`_compute_error`) and adjustment (`_adaptive_step`) based on integrator order, with configurable tolerances (atol, rtol), safety factor, and min/max dt bounds for improved accuracy and efficiency in dynamic systems.
- **Momentum Conservation Tracking:** Added logging and computation of linear and angular momentum history in `simulation.py` via `_compute_momenta`, relative to the center of mass, to verify conservation properties.
- **State Vector Management:** Added helper methods `_pack_state` and `_unpack_state` in `simulation.py` for flattening/unflattening positions and velocities, enabling seamless integration with ODE solvers.
- **Integrator Order Mapping:** Defined an `order_map` in `simulation.py` to support adaptive stepping by associating numerical orders with integrators (e.g., Euler: 1, RK4: 4, Velocity Verlet: 2).
- **Rotational Placeholders:** Added optional `orientation` (quaternion) and `angular_vel` parameters to the `Body` class in `body.py` as placeholders for future rotational dynamics (currently commented out with validation).
- **3D Trajectory Logging:** Enhanced trajectory storage in `simulation.py` to capture full 3D positions for all bodies, supporting advanced visualizations like projections.

### Changed
- **Body Class Simplification:** Removed the `Earth` subclass from `body.py`; now all bodies (e.g., Sun, Earth) are instantiated as generic `Body` objects for greater flexibility.
- **Integrator Refactoring:** Updated integrators in `integrators.py` to operate on flattened state vectors instead of direct Body mutations, improving modularity and compatibility with adaptive stepping.
- **Simulation Initialization:** Added parameters for adaptive control (atol, rtol, safety, min_dt, max_dt) in `Simulation__init__` for finer simulation tuning.
- **Main Script Example:** Shifted the demonstration in `main.py` from a short Earth-meteor collision to a year-long Sun-Earth orbital simulation, integrating the new `Analysis` class for comprehensive result evaluation (e.g., energy drift rate, projections, momentum plots).
- **Plotting Decoupling:** Moved all visualization logic from `simulation.py` to the new `Analysis` class, promoting separation of concerns.
- **Docstring Enhancements:** Expanded docstrings across all modules for better clarity, including notes on usage, limitations, and future extensions (e.g., symplectic integrators for energy conservation).
- **Collision and Energy Handling:** Minor refinements in `_handle_collisions` and `_compute_energy` for consistency with state vector approach.

### Removed
- **In-Place Body Updates in Integrators:** Eliminated direct mutations in old integrator functions to favor functional state returns.

---

## [v2.5]
### Added
- **Velocity Verlet Integration:** Implemented in `integrators.py` for improved numerical stability and accuracy in simulating particle trajectories.

### Changed
- **Code Architecture:** Refactored computational and structural organization for improved readability and maintainability.
- **RK4 Update:** Avoided direct mutation of body states during intermediate `RK4 steps`; redesigned `rk4` and `forces.py` accordingly.
- **Force Function Update:** Modified `forces.py` functions to accept masses and positions instead of Body objects, reducing computational overhead.
- **Simulation Enhancements:** Added new helper methods in `simulation.py` to streamline simulation workflow and modularity.

---

## [v2.0]
### Added
- **Universal Gravitation:** Replaced flat-earth gravity with Newton's Law of Universal Gravitation ($F = G \frac{m_1 m_2}{r^2}$) for N-body simulation capability.
- **Planetary Physics:** Added `Earth` class inheriting from `Body` with standard planetary mass ($5.97 \times 10^{24}$ kg) and radius.
- **Energy Analytics:** Implemented real-time tracking of Kinetic, Potential, and Total Energy to verify simulation stability.
- **Data Visualization:** Added `matplotlib` integration to generate post-simulation plots for Speed vs. Time and Energy conservation.
- **Collision Logic:** Enhanced `_resolve_collision` with both impulse-based momentum transfer and positional correction to prevent body interpenetration.

### Changed
- **Modular Architecture:** Refactored monolithic codebase into dedicated modules (`body.py`, `forces.py`, `integrators.py`, `simulation.py`) for better maintainability.
- **Integration Method:** Standardized RK4 implementation with a helper `derivative` function for cleaner state vector management.
- **Body Definition:** Updated `Body` class to require a `radius` parameter, enabling geometric collision detection.

---

## [v1.0]
### Added
- Initial release of the engine.
- Euler and Runge-Kutta 4 (RK4) numerical integration methods.
- Basic particle physics with mass, position, and velocity.
- Simple ground collision detection with coefficient of restitution.
- Hardcoded gravity force.
