# Limitations

## Overview

Neutral Physics Engine implements a variety of techniques for numerical simulation, spatial acceleration, collision detection, and scientific data management. While the current architecture is capable of handling a wide range of gravitational simulations, several limitations remain.

Many of these limitations are intentional tradeoffs made to preserve clarity, maintainability, and architectural simplicity.

---

# Single-Threaded Execution

The engine currently executes its computational workload on a single CPU thread.

Although the Barnes-Hut algorithm significantly reduces computational complexity, simulation performance remains constrained by single-core execution.

Potential improvements include parallel tree construction, parallel traversal, and multithreaded force evaluation.

---

# No Explicit SIMD Optimization

The native backend relies primarily on compiler optimizations rather than manually implemented SIMD instructions.

Modern processors support vectorized floating-point operations that can evaluate multiple calculations simultaneously.

The current implementation does not explicitly leverage these capabilities.

As a result, additional performance may be available through low-level vectorization techniques.

---

# CPU-Only Computation

All simulation calculations are currently performed on the CPU.

While modern CPUs are highly capable, massively parallel workloads can often benefit from specialized hardware acceleration.

The current architecture does not include support for GPU-based force evaluation or collision processing.

---

# Barnes-Hut Approximation Error

The Barnes-Hut algorithm improves performance by approximating distant groups of bodies using aggregated center-of-mass representations.

This introduces approximation error that is not present in direct pairwise force calculations.

The magnitude of this error depends on the selected opening-angle parameter (`theta`).

Users must balance computational performance against force accuracy according to their simulation requirements.

---

# Limited Physical Models

The engine focuses primarily on Newtonian gravitational dynamics and spherical collision handling.

Several classes of physical phenomena are outside the scope of the current implementation, including:

* Electromagnetic interactions
* Relativistic effects
* Deformable materials
* Fluid dynamics
* Thermal systems

The engine should therefore be viewed as a specialized gravitational simulation framework rather than a general-purpose physics engine.

---

# Rotational Dynamics

The current implementation models translational motion only.

Although placeholders for rotational state representations were explored during development, complete rotational dynamics are not implemented.

Missing capabilities include:

* Torque calculations
* Rotational integration
* Inertia tensors
* Angular collision response

Bodies are therefore treated as objects whose motion is fully described by position and velocity.

---

# Memory Requirements for Large Simulations

The introduction of HDF5 telemetry significantly reduced memory consumption during long simulation runs.

However, large simulations still require substantial memory for:

* Body storage
* State vectors
* Octree nodes
* Temporary numerical buffers

As body counts continue to increase, memory consumption becomes an important consideration.

---

# Python Orchestration Overhead

The computationally intensive portions of the engine execute in C++, but simulation orchestration remains in Python.

This hybrid architecture provides flexibility and rapid development but introduces some unavoidable interpreter overhead.

While this overhead is small compared to force evaluation costs, it remains present in the overall execution pipeline.

---

# Numerical Precision

All numerical simulations are subject to floating-point limitations.

Sources of numerical error include:

* Rounding error
* Truncation error
* Approximation error
* Finite precision arithmetic

Although adaptive stepping, modern integrators, and diagnostic tools help control these effects, they cannot be eliminated entirely.

Users should interpret results within the context of these numerical limitations.

---

# Validation Scope

The engine includes automated tests, benchmarking tools, and conservation diagnostics.

However, the project has not been validated against every possible simulation scenario.

Users are encouraged to verify results when applying the engine to new problems, especially those involving large scales, unusual initial conditions, or extreme parameter values.

---

# Summary

The current architecture successfully balances performance, modularity, and numerical capability for gravitational simulation workloads.

At the same time, several limitations remain in areas such as parallelism, hardware utilization, physical modeling, and numerical precision.

Understanding these limitations is important for interpreting results correctly and for identifying areas where the engine may be extended or refined in the future.
