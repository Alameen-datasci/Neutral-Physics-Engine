# Neutral Physics Engine

A modular, first-principles N-Body physics engine written in Python.

## Overview
Neutral Physics Engine is a custom simulation framework designed to model gravitational interactions and rigid body dynamics. It uses numerical integration methods (RK4) to solve equations of motion for complex systems, ranging from simple falling objects to planetary orbits.

The engine is built with a focus on code readability, modularity, and mathematical accuracy, making it suitable for educational demonstrations and portfolio showcases.

## Features (v2.0)
- **N-Body Gravitation:** Simulates mutual gravitational forces between all objects using Newton's Law of Universal Gravitation.
- **Numerical Integration:** Implements custom **Runge-Kutta 4 (RK4)** and Euler solvers for high-precision time-stepping.
- **Energy Conservation:** Tracks Kinetic, Potential, and Total Energy to verify simulation stability.
- **Collision Resolution:** Handles inelastic collisions with coefficient of restitution and positional correction (preventing object overlap).
- **Visualization:** Integrated `matplotlib` plotting for trajectory analysis and energy auditing.