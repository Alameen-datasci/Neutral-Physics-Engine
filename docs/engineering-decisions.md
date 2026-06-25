# Engineering Decisions

This document describes the major engineering decisions made throughout the development of Neutral Physics Engine, including the problems encountered, alternatives considered, and the reasoning behind each solution.

---

# Numerical Integrators

## Problem

The first version of the engine used only the Euler integrator because of its simplicity and ease of implementation.

However, after studying numerical integration methods, it became clear that Euler introduces significant numerical error over time. In conservative systems such as orbital mechanics and simple harmonic motion, the method artificially injects or removes energy from the system, causing trajectories to diverge from physically realistic behavior.

## Investigation

To understand this issue, I derived the energy evolution of a simple harmonic oscillator using Euler integration and observed the appearance of additional error terms that are not present in the analytical solution.

This motivated a deeper study of numerical integration theory.

## Decision

The Runge-Kutta 4 (RK4) method was added as a higher-accuracy alternative.

Later, Velocity Verlet was introduced as a symplectic integrator specifically designed for Hamiltonian systems.

## Tradeoffs

Euler:

* Extremely simple
* Fast
* Poor long-term accuracy

RK4:

* High local accuracy
* Higher computational cost
* Accumulated energy drift during long simulations

Velocity Verlet:

* Lower computational cost than RK4
* Excellent long-term energy behavior
* Particularly well suited for orbital mechanics

## Result

The engine now supports multiple integration strategies, allowing users to balance computational cost and numerical accuracy depending on the simulation scenario.

---

# Barnes-Hut Spatial Acceleration

## Problem

The original gravitational implementation evaluated every pairwise interaction directly.

This scales as O(N²), making large simulations increasingly expensive as body counts grow.

## Investigation

Several spatial acceleration techniques were researched, including quadtrees and the Barnes-Hut algorithm.

Because the engine operates in three-dimensional space, an octree-based structure was required rather than a quadtree.

## Decision

The Barnes-Hut algorithm was implemented using an octree data structure.

Distant particle groups are approximated using center-of-mass aggregation rather than computing every interaction directly.

## Tradeoffs

Advantages:

* Approximately O(N log N) scaling
* Supports significantly larger simulations

Disadvantages:

* Introduces approximation error
* Requires additional tree construction overhead

## Result

Barnes-Hut became the foundation of the engine's gravitational and collision systems and enabled simulations that would be impractical using direct summation.

---

# Native C++ Backend

## Problem

Although Barnes-Hut improved algorithmic complexity, profiling revealed that tree construction and traversal remained the dominant runtime cost.

Recursive node traversal and object management in Python introduced significant overhead.

## Investigation

Performance profiling indicated that the octree implementation was the primary computational hotspot.

A native implementation was explored to eliminate interpreter overhead.

## Decision

The Barnes-Hut octree was rewritten in C++17 and exposed to Python through Pybind11.

Aggressive compiler optimizations such as O3, fast-math, and architecture-specific optimizations were enabled during compilation.

## Tradeoffs

Advantages:

* Significant execution speedup
* Reduced interpreter overhead
* Better utilization of CPU optimizations

Disadvantages:

* Increased build complexity
* Additional maintenance burden due to mixed-language architecture

## Result

The C++ backend became the computational core of the engine and produced substantial performance improvements compared to the original Python implementation.

---

# HDF5 Telemetry Pipeline

## Problem

Long simulations generate large amounts of telemetry data.

Keeping complete histories in memory causes RAM consumption to grow continuously with simulation duration.

## Investigation

Several storage approaches were considered.

Traditional text-based formats such as CSV were unsuitable due to storage overhead and inefficient read/write performance.

## Decision

The engine adopted HDF5 as its primary telemetry format.

Simulation data is buffered in memory and periodically written to compressed HDF5 datasets.

## Tradeoffs

Advantages:

* Reduced memory consumption
* Efficient storage of large datasets
* Metadata support
* Fast partial reads during analysis

Disadvantages:

* More complex implementation than simple text formats
* Requires external dependencies

## Result

The telemetry pipeline became capable of handling significantly larger simulations without exhausting system memory.

---

# Adaptive Time Stepping

## Problem

A fixed timestep must be chosen conservatively to handle the most demanding regions of a simulation.

This often wastes computation when the system evolves slowly.

## Decision

Adaptive timestep control was introduced using local error estimation.

The timestep dynamically expands or contracts depending on the estimated integration error.

## Result

The engine can automatically balance accuracy and performance without requiring manual timestep tuning for every scenario.

---

# Zero-Copy Python–C++ Interface

## Problem

Even after moving the octree to C++, transferring data between NumPy arrays and C++ structures introduced avoidable overhead.

## Decision

The Python bindings were redesigned to operate directly on NumPy memory buffers.

This removed unnecessary copies during Python-to-C++ communication.

## Result

Initialization overhead was reduced and data transfer between Python and C++ became significantly more efficient.

---

# Design Philosophy

Throughout development, the primary engineering goal was not simply adding features, but understanding the computational and numerical limitations of each implementation and replacing them with more scalable alternatives.

Many of the architectural decisions in Neutral Physics Engine originated from encountering a bottleneck, studying the underlying theory, and then redesigning the system around a more appropriate solution.
