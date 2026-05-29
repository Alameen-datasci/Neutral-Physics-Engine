/**
 * @file pybind11_wrapper.cpp
 * @brief Pybind11 bindings for the Barnes-Hut Octree C++ implementation.
 * * This module exposes the high-performance C++ Octree class to Python. 
 * It handles the memory bridging between Python NumPy arrays and C++ 
 * standard vectors, allowing the physics engine to leverage C++ execution speeds 
 * without altering the Python-side API.
 */
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include "octree.cpp"

namespace py = pybind11;

// Define the Python module name (octree_ext)
PYBIND11_MODULE(octree, m) {
    py::class_<Octree>(m, "Octree")

    /**
         * @brief Custom Constructor Binding
         * * Intercepts NumPy arrays (`py::array_t`) from Python, extracts their 
         * underlying raw memory pointers, and constructs the necessary C++ 
         * `std::vector` objects to feed into the actual C++ Octree constructor.
         */
    .def(py::init([](
        py::array_t<double> masses_in,
        py::array_t<double> pos_in,
        std::optional<py::array_t<double>> radii_in,
        double theta) {
            // Request access to the raw memory buffers of the NumPy arrays
            auto buf_mass = masses_in.request();
            auto buf_pos = pos_in.request();

            // Determine the number of bodies (N) using size_t to prevent signed/unsigned warnings
            size_t N = static_cast<size_t>(buf_mass.shape[0]);

            // Extract typed pointers to the start of the memory buffers
            double* ptr_mass = static_cast<double*>(buf_mass.ptr);
            double* ptr_pos = static_cast<double*>(buf_pos.ptr);

            // Construct the masses vector directly from the memory pointer
            std::vector<double> masses(ptr_mass, ptr_mass + N);

            // Manually unpack the flattened (N*3) position array into a vector of vec3 structs
            std::vector<vec3> positions;
            positions.reserve(N); // Pre-allocate memory for speed
            for (size_t i = 0; i < N; i++) {
                positions.push_back({ptr_pos[i*3], ptr_pos[i*3 + 1], ptr_pos[i*3 + 2]});
            }

            // Handle optional radii array for collision detection
            std::optional<std::vector<double>> radii_opt = std::nullopt;
            if (radii_in) {
                auto buf_radii = radii_in->request();
                double* ptr_radii = static_cast<double*>(buf_radii.ptr);
                radii_opt = std::vector<double>(ptr_radii, ptr_radii + N);
            }

            return std::make_unique<Octree>(masses, positions, radii_opt, theta);
        }),
        py::arg("masses"),
        py::arg("pos"),
        py::arg("radii") = py::none(), // Optional argument with default value of None
        py::arg("theta") = 0.5) // Default value for theta

        // Expose standard methods directly to Python
        .def("build", &Octree::build, "Build the octree from the current position state.")
        .def("find_collisions", &Octree::find_collisions, "Find all intersecting body pairs in O(N log N) time.")
        .def("compute_potential", &Octree::compute_potential, "Compute the gravitational potential for a specific body.")
        
        /**
         * @brief Lambda binding for compute_acceleration
         * * Unpacks the custom `vec3` struct returned by C++ into a standard 
         * Python tuple (`py::make_tuple`) so it can be understood by the Python runtime.
         */
        .def("compute_acceleration", [](Octree& self, int i) {
            vec3 acc = self.compute_acceleration(i);
            return py::make_tuple(acc.x, acc.y, acc.z);
        }, "Compute the approximate acceleration vector for a specific body.")

        /**
         * @brief Lambda binding for compare_with_direct
         * * Unpacks the nested C++ tuple and custom `vec3` structs into 
         * standard nested Python tuples.
         */
        .def("compare_with_direct", [](Octree& self, int i) {
            auto [tree_acc, direct_acc, error] = self.compare_with_direct(i);
            return py::make_tuple(
                py::make_tuple(tree_acc.x, tree_acc.y, tree_acc.z),
                py::make_tuple(direct_acc.x, direct_acc.y, direct_acc.z),
                error
            );
        }, "Compare the tree-approximated acceleration with direct O(N^2) summation.");
}