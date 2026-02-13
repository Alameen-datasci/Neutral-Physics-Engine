# Changelog

All notable changes to the "Neutral Physics Engine" project will be documented in this file.

## [2.0]
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

## [1.0]
### Added
- Initial release of the engine.
- Euler and Runge-Kutta 4 (RK4) numerical integration methods.
- Basic particle physics with mass, position, and velocity.
- Simple ground collision detection with coefficient of restitution.
- Hardcoded gravity force.
