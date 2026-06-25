# Performance Analysis

## Overview

Performance optimization became a major focus throughout the development of Neutral Physics Engine. While early versions were sufficient for small demonstrations, increasing body counts quickly exposed computational and memory bottlenecks.

The optimization process followed three primary goals:

1. Reduce computational complexity.
2. Reduce Python runtime overhead.
3. Improve memory and storage efficiency.

This document summarizes the major performance improvements introduced during development.

---

# Baseline Implementation

The earliest versions of the engine used direct pairwise gravitational force calculations.

For every body, the engine computed interactions with every other body:

```text
for body_i:
    for body_j:
        compute_gravity()
```

This approach has a computational complexity of:

```text
O(N²)
```

where N is the number of bodies.

Although simple and mathematically exact, the number of force evaluations grows rapidly as body counts increase.

| Bodies | Pairwise Interactions |
| ------ | --------------------- |
| 100    | 10,000                |
| 1,000  | 1,000,000             |
| 10,000 | 100,000,000           |

As simulations scaled, force evaluation became the dominant runtime bottleneck.

---

# Barnes-Hut Spatial Acceleration

## Motivation

Direct summation does not scale well for large N-body systems.

To reduce the computational burden, the engine adopted the Barnes-Hut algorithm using an octree-based spatial partitioning structure.

## Approach

Instead of evaluating every interaction directly, distant groups of particles are approximated using a single center-of-mass representation.

The octree recursively subdivides space into smaller regions and stores aggregate mass information for each node.

## Complexity Improvement

| Method            | Complexity |
| ----------------- | ---------- |
| Direct Summation  | O(N²)      |
| Barnes-Hut Octree | O(N log N) |

This dramatically improves scalability as body counts increase.

## Tradeoff

The Barnes-Hut approximation introduces force error controlled by the opening-angle parameter:

```text
theta
```

Smaller values increase accuracy while increasing computational cost.

Larger values increase performance while sacrificing accuracy.

Benchmarking tools were developed to quantify this tradeoff.

---

# Python Performance Bottlenecks

After Barnes-Hut was implemented, algorithmic complexity was no longer the primary limitation.

Profiling revealed that Python interpreter overhead became a significant contributor to runtime.

The most expensive operations included:

* Recursive octree traversal
* Node allocation
* Object creation
* Function call overhead

Although the algorithm scaled efficiently, Python execution remained a bottleneck.

---

# Native C++ Backend

## Motivation

Tree construction and traversal are executed repeatedly during every simulation step.

These operations are highly recursive and computationally intensive.

## Solution

The Barnes-Hut implementation was rewritten in C++17 and exposed to Python using Pybind11.

The native backend performs:

* Octree construction
* Center-of-mass aggregation
* Tree traversal
* Force evaluation
* Broad-phase collision queries

while Python remains responsible for orchestration and user interaction.

## Compiler Optimization

The build system automatically enables aggressive optimization flags where available:

```text
-O3
-ffast-math
-march=native
```

These optimizations allow the compiler to generate architecture-specific machine code and exploit available CPU instruction sets.

## Result

The C++ implementation reduced interpreter overhead and delivered substantial performance improvements compared to the original Python version.

The Barnes-Hut algorithm remained unchanged mathematically; only the execution layer was optimized.

---

# Memory Optimization

## **slots**

The `Body` and `Node` classes use:

```python
__slots__
```

instead of standard Python instance dictionaries.

### Benefits

* Reduced memory consumption per object
* Faster attribute access
* Improved cache locality

This becomes increasingly important when constructing large octrees containing thousands of nodes.

---

## Allocation Reduction

Several frequently executed code paths were redesigned to minimize temporary object creation.

Examples include:

* Reusing preallocated acceleration arrays
* In-place state updates
* Avoiding repeated array construction

Reducing allocations decreases garbage collection pressure and improves runtime consistency.

---

# Mathematical Hot-Loop Optimization

Many numerical operations originally relied on high-level NumPy calls.

Examples included:

```python
np.linalg.norm()
np.subtract()
```

Although efficient for large arrays, these functions introduce overhead when repeatedly operating on small three-dimensional vectors.

The engine replaced many of these calls with explicit scalar operations:

```python
dx*dx + dy*dy + dz*dz
```

Benefits include:

* Reduced function dispatch overhead
* Fewer temporary arrays
* Faster execution inside heavily repeated loops

---

# Zero-Copy Python–C++ Interface

## Problem

Even after moving the octree to C++, transferring data between NumPy arrays and C++ structures introduced unnecessary overhead.

The original implementation copied data into C++ containers before processing.

## Solution

The interface was redesigned to operate directly on NumPy memory buffers.

Raw pointers are passed directly into the native backend.

## Benefits

* Eliminated initialization copies
* Reduced allocation overhead
* Faster Python-to-C++ communication

This optimization significantly reduced the cost of crossing language boundaries.

---

# HDF5 Telemetry Pipeline

## Problem

Long simulations generate large amounts of telemetry:

* Positions
* Velocities
* Energies
* Momenta

Keeping all history in memory causes RAM usage to grow continuously.

## Solution

The engine stores telemetry using HDF5.

Data is buffered in memory and periodically flushed to disk using:

* Chunked storage
* Compression
* Extendable datasets

## Benefits

* Reduced memory consumption
* Efficient storage of large simulations
* Fast analysis without loading entire datasets

---

# Benchmarking Infrastructure

To verify performance claims, a dedicated benchmarking suite was introduced.

## Scaling Analysis

Measures:

* Runtime scaling
* Barnes-Hut complexity behavior
* Python versus C++ implementations

## Theta Tradeoff Analysis

Measures:

* Force approximation error
* Runtime versus accuracy tradeoffs

## Collision Stress Testing

Measures:

* Broad-phase collision performance
* Scaling under increasing body counts

## HDF5 Performance Testing

Measures:

* Storage overhead
* Buffer size effects
* Disk throughput

## Integrator Comparisons

Measures:

* Energy drift
* Runtime cost
* Numerical stability

---

# Current Limitations

Despite substantial optimization work, several opportunities remain for future performance improvements.

Current limitations include:

* Single-threaded force evaluation
* No OpenMP parallelization
* No SIMD intrinsics
* No GPU acceleration
* Python orchestration layer remains present
* Single-node execution only

These areas represent the next stage of development beyond the current architecture.

---

# Summary

Performance improvements in Neutral Physics Engine came from both algorithmic and implementation-level optimization.

Major improvements included:

* O(N²) → O(N log N) Barnes-Hut acceleration
* Native C++ octree backend
* Zero-copy NumPy integration
* Memory optimization using `__slots__`
* Allocation reduction
* Hot-loop scalar math optimization
* Buffered HDF5 telemetry storage

The result is an engine capable of handling substantially larger simulations than the original implementation while maintaining a modular architecture suitable for scientific computing experimentation.
