"""
Build script for compiling the high-performance C++ Octree extension.

This script uses setuptools and pybind11 to compile the C++ source files
into a native Python module (.so on Unix, .pyd on Windows). It automatically
applies aggressive, platform-specific compiler optimizations to maximize
the execution speed of the physics engine.
"""
import sys
from setuptools import setup

# Attempt to import pybind11 build helpers, failing gracefully with an exit code if missing
try:
    from pybind11.setup_helpers import Pybind11Extension, build_ext
except ImportError:
    print("Error: pybind11 is not installed. Run 'pip install pybind11' first.")
    sys.exit(1) # Exit Code 1 indicates a fatal error to the operating system

# Detect platform and apply appropriate compiler optimization flags
if sys.platform == "win32":
    # MSVC flags: /O2 (Maximize Speed), /fp:fast (Fast floating-point math)
    compile_args = ["/O2", "/fp:fast"]
else:
    # GCC/Clang flags for macOS/Linux:
    # -O3: Maximum compiler optimization level
    # -ffast-math: Aggressive floating point math shortcuts (prioritizes speed over strict IEEE compliance)
    # -march=native: Unlocks CPU-specific hardware instructions for maximum performance on the host machine
    compile_args = ["-O3", "-ffast-math", "-march=native"]

# Define the C++ extension module
ext_modules = [
    Pybind11Extension(
        "neutral_physics_engine.octree",                 # Target module name and package path
        ["src/neutral_physics_engine/pybind11_wrapper.cpp"], # Relative path to the wrapper source file
        cxx_std=17,                                          # Enforce C++17 standard (required for std::optional)
        extra_compile_args=compile_args                      # Inject the speed optimization flags
    ),
]

# Execute the setup
setup(
    name="neutral-physics-engine",
    version="4.2.0",
    description="High-performance Barnes-Hut Octree via Pybind11",
    ext_modules=ext_modules,
    cmdclass={"build_ext": build_ext},
    zip_safe=False, # Force installation as an unzipped folder so the OS linker can access the compiled binary
)