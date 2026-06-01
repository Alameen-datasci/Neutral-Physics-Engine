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

// Define the Python module name
PYBIND11_MODULE(octree, m) {

    py::class_<vec3>(m, "vec3")
        .def_readonly("x", &vec3::x)
        .def_readonly("y", &vec3::y)
        .def_readonly("z", &vec3::z);

    // Expose the Node struct
    py::class_<Node>(m, "Node")
        .def_readonly("mass", &Node::mass)
        .def_readonly("center", &Node::center)
        .def_readonly("half_side", &Node::half_side)
        .def_readonly("com", &Node::com)
        .def_readonly("body_index", &Node::body_index)
        .def_readonly("max_radius", &Node::max_radius);


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

            // Determine the number of bodies (N) using size_t to prevent signed/unsigned warnings
            size_t N = static_cast<size_t>(buf_mass.shape[0]);

            // Extract typed pointers to the start of the memory buffers
            const double* ptr_mass = static_cast<const double*>(buf_mass.ptr);
            const double* raw_ptr_pos = static_cast<const double*>(pos_in.request().ptr);

            // Tell C++ to view the raw double array as an array of vec3s
            const vec3* ptr_pos = reinterpret_cast<const vec3*>(raw_ptr_pos);

            // Handle optional radii
            const double* ptr_radii = nullptr;
            if (radii_in) {
                ptr_radii = static_cast<const double*>(radii_in->request().ptr);
            }

            // Pass pointers directly
            return std::make_unique<Octree>(ptr_mass, ptr_pos, N, ptr_radii, theta);
        }),
        py::arg("masses"),
        py::arg("pos"),
        py::arg("radii") = py::none(), // Optional argument with default value of None
        py::arg("theta") = 0.5) // Default value for theta

        .def_property_readonly("root", &Octree::get_root, "Get the root node of the octree")

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