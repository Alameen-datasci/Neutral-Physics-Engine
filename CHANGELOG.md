# Changelog

All notable changes to the "Neutral Physics Engine" project will be documented in this file.

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

## [v2.5]
### Added
- **Velocity Verlet Integration:** Implemented in `integrators.py` for improved numerical stability and accuracy in simulating particle trajectories.

### Changed
- **Code Architecture:** Refactored computational and structural organization for improved readability and maintainability.
- **RK4 Update:** Avoided direct mutation of body states during intermediate `RK4 steps`; redesigned `rk4` and `forces.py` accordingly.
- **Force Function Update:** Modified `forces.py` functions to accept masses and positions instead of Body objects, reducing computational overhead.
- **Simulation Enhancements:** Added new helper methods in `simulation.py` to streamline simulation workflow and modularity.


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

## [v1.0]
### Added
- Initial release of the engine.
- Euler and Runge-Kutta 4 (RK4) numerical integration methods.
- Basic particle physics with mass, position, and velocity.
- Simple ground collision detection with coefficient of restitution.
- Hardcoded gravity force.
