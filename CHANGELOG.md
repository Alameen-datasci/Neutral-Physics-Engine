# Changelog

All notable changes to the "Neutral Physics Engine" project will be documented in this file.

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
- **Initial Engine Release:** Core framework for simulating particle physics.
- **Flat-Earth Gravity Model:** Simplified uniform gravitational acceleration.
- **Numerical Integrators:** `Euler` and `Runge-Kutta 4 (RK4)` methods for particle state updates.
- **Basic Particle Physics:** Bodies with mass, position, and velocity attributes.
- **Collision Detection:** Simple ground collision with coefficient of restitution and basic jitter handling.
- **Hardcoded Forces:** Gravity applied as a fixed downward force on particles.