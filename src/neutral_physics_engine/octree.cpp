#include <array>
#include <stdexcept>
#include <memory>
#include <vector>
#include <optional>
#include <utility>
#include <algorithm>
#include <cmath>
#include <tuple>


constexpr double G = 6.67430e-11;
constexpr double EPS = 1e-10;
constexpr double eps2 = EPS * EPS;

struct vec3
{
    double x, y, z;
};
    
// mathematical operations

/* Vector Addition */
vec3 operator+(const vec3& a, const vec3& b) {
    return {
        a.x + b.x,
        a.y + b.y,
        a.z + b.z
    };
}

/* Vector Subtraction */
vec3 operator-(const vec3& a, const vec3& b) {
    return {
        a.x - b.x,
        a.y - b.y,
        a.z - b.z
    };
}

/* Vector Multiplication with scalar */
vec3 operator*(double k, const vec3& v) {
    return {k * v.x, k * v.y, k * v.z};
}

vec3 operator*(const vec3& v, double k) {
    return {v.x * k, v.y * k, v.z * k};
}

/* Vector Division with scalar */
vec3 operator/(const vec3& v, double k) {
    return {v.x / k, v.y / k, v.z / k};
}

// Dot Product of Vectors
double dot(const vec3& a, const vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

// Cross Product of Vectors
vec3 cross(const vec3& a, const vec3& b) {
    return {
        a.y * b.z - b.y * a.z,
        a.z * b.x - a.x * b.z,
        a.x * b.y - b.x * a.y
    };
}

// Vector Norm
double norm(const vec3& v) {
    return std::sqrt(dot(v, v));
}

// Node
struct Node
{
    vec3 center;
    double half_side;
    int body_index = -1;
    std::array<std::unique_ptr<Node>, 8> children;
    vec3 com = {0.0, 0.0, 0.0};
    double mass = 0.0;
    double max_radius = 0.0;

    // Constructor
    Node(const vec3& c, const double hs)
    : center(c), half_side(hs)
    {
        if (hs <= 0) {
            throw std::invalid_argument("Half Side should be Positive.");
        }
    }
};

// Octree
class Octree {
    private:
        const double* masses;     // Pointer to Numpy masses array [N]
        const vec3* pos;          // Pointer to Numpy N x 3 pos array (cast to vec3)
        std::unique_ptr<Node> root;
        size_t num_bodies;          // Number of bodies (size of masses and pos arrays)
        // const std::optional<std::reference_wrapper<const std::vector<double>>> radii;
        const double* radii;         // Pointer to Numpy radii array [N], optional (can be nullptr)
        double theta = 0.5;
        std::vector<std::pair<int, int>> collision_pairs;

    public:
        //Constructor
        Octree(
            const double* masses_,
            const vec3* pos_,
            const size_t num_bodies_,
            const double* radii_ = nullptr,
            const double theta_ = 0.5
        ) : masses(masses_), pos(pos_), num_bodies(num_bodies_), radii(radii_), theta(theta_)
        {
            if (num_bodies == 0 || pos == nullptr) {
                throw std::invalid_argument("Cannot build octree: positions array is empty or null.");
            }
        }

        const Node* get_root() const {
            return root.get();
        }

        // build(): building the octree
        void build() {
            // initially take first body's position as bounding box
            double xmin = pos[0].x, ymin = pos[0].y, zmin = pos[0].z;
            double xmax = pos[0].x, ymax = pos[0].y, zmax = pos[0].z;

            for (size_t i = 1; i < num_bodies; i++) {
                const auto& p = pos[i];

                if (p.x < xmin) xmin = p.x;
                if (p.y < ymin) ymin = p.y;
                if (p.z < zmin) zmin = p.z;

                if (p.x > xmax) xmax = p.x;
                if (p.y > ymax) ymax = p.y;
                if (p.z > zmax) zmax = p.z;
            }

            double half_side = std::max({xmax - xmin, ymax - ymin, zmax - zmin}) * 0.5 + 1e-5;
            
            double cx = (xmax + xmin) * 0.5, cy = (ymax + ymin) * 0.5, cz = (zmax + zmin) * 0.5;

            // create initial Node or create root
            vec3 center{cx, cy, cz};
            root = std::make_unique<Node>(center, half_side);

            // insert bodies
            for (size_t i = 0; i < num_bodies; i++) {
                insert(i);
            }
        }

        void insert(int i) {
            if (!root) throw std::invalid_argument("Call build() before insert().");

            insert_impl(root.get(), i);
        }

        void subdivide(Node* node) {
            double child_half = node->half_side * 0.5;

            int idx = 0;

            for (double dx : {-child_half, child_half}) {
                for (double dy : {-child_half, child_half}) {
                    for (double dz : {-child_half, child_half}) {
                        vec3 child_center{
                            node->center.x + dx,
                            node->center.y + dy,
                            node->center.z + dz
                        };

                        node->children[idx] = std::make_unique<Node>(child_center, child_half);
                        idx++;
                    }
                }
            }
        }

        Node* choose_child(Node* node, const vec3& pos) {
            int idx = get_octant_index(node, pos);
            return node->children[idx].get();
        }

        int get_octant_index(Node* node, const vec3& pos) {
            int ix = (pos.x >= node->center.x) ? 1 : 0;
            int iy = (pos.y >= node->center.y) ? 1 : 0;
            int iz = (pos.z >= node->center.z) ? 1 : 0;
            return ix * 4 + iy * 2 + iz;
        }

        vec3 compute_acceleration(int i) {
            vec3 acc{0.0, 0.0, 0.0};
            compute_node_forces(root.get(), i, acc);
            return acc;
        }

        std::tuple<vec3, vec3, double> compare_with_direct(int i) {
            vec3 tree_acc = compute_acceleration(i);
            vec3 direct_acc{0.0, 0.0, 0.0};

            for (size_t j = 0; j < num_bodies; j++) {
                if (static_cast<size_t>(i) == j) continue;

                double dx = pos[j].x - pos[i].x, dy = pos[j].y - pos[i].y, dz = pos[j].z - pos[i].z;
                double inv_r = 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz + eps2);
                double inv_r3 = inv_r * inv_r * inv_r;
                double factor = G * masses[j] * inv_r3;

                direct_acc.x += factor * dx, direct_acc.y += factor * dy, direct_acc.z += factor * dz;
            }

            double error = norm(tree_acc - direct_acc);
            return {tree_acc, direct_acc, error};
        }

        double point_to_node_distance(Node* node, const vec3& pos) {
            double dx, dy, dz;
            if (pos.x < (node->center.x - node->half_side)) dx = (node->center.x - node->half_side) - pos.x;
            else if (pos.x > (node->center.x + node->half_side)) dx = pos.x - (node->center.x + node->half_side);
            else dx = 0.0;

            if (pos.y < (node->center.y - node->half_side)) dy = (node->center.y - node->half_side) - pos.y;
            else if (pos.y > (node->center.y + node->half_side)) dy = pos.y - (node->center.y + node->half_side);
            else dy = 0.0;

            if (pos.z < (node->center.z - node->half_side)) dz = (node->center.z - node->half_side) - pos.z;
            else if (pos.z > (node->center.z + node->half_side)) dz = pos.z - (node->center.z + node->half_side);
            else dz = 0.0;

            return std::sqrt(dx*dx + dy*dy + dz*dz);
        } 

        std::vector<std::pair<int, int>> find_collisions() {
            if (!root) throw std::invalid_argument("Build octree before collision detection.");

            if (!radii) throw std::invalid_argument("Radii are required for collision detection.");

            collision_pairs.clear();

            for (size_t i = 0; i < num_bodies; i++) {
                traverse_for_collisions(root.get(), i);
            }

            return collision_pairs;
        }

        double compute_potential(int i) {
            if (!root) throw std::invalid_argument("call build() first.");
            return compute_node_potential(root.get(), i);
        }

    private:
        void insert_impl(Node* node, int i) {
            double old_mass = node->mass;
            double new_mass = old_mass + masses[i];
            const vec3& r = pos[i];

            if (old_mass == 0) {
                node->com = r;
            } else {
                node->com = (node->com * old_mass + r * masses[i]) / new_mass;
            }
            node->mass = new_mass;

            if (radii) {
                node->max_radius = std::max(node->max_radius, radii[i]);
            }

            // case one: Empty Leaf
            if (node->body_index == -1 && node->children[0] == nullptr) {
                node->body_index = i;
                return;
            }

            // case two: Leaf with One Body
            else if (node->body_index != -1 && node->children[0] == nullptr)
            {
                int old_body_index = node->body_index;

                node->body_index = -1;

                subdivide(node);

                insert_impl(choose_child(node, pos[old_body_index]), old_body_index);

                insert_impl(choose_child(node, pos[i]), i);

                return;
            }

            // case three: Internal Node
            else {
                insert_impl(choose_child(node, pos[i]), i);
            }
            
        }

        void compute_node_forces(Node* node, int i, vec3& acc) {
            if (node->mass < EPS) return;

            if (node->children[0] == nullptr && node->body_index == i) return;

            if (node->children[0] == nullptr && node->body_index != i) {
                int j = node->body_index;
                double dx = pos[j].x - pos[i].x, dy = pos[j].y - pos[i].y, dz = pos[j].z - pos[i].z;
                double inv_dist = 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz + eps2);
                double inv_dist3 = inv_dist * inv_dist * inv_dist;
                double f_common = G * masses[j] * inv_dist3;

                acc.x += f_common * dx, acc.y += f_common * dy, acc.z += f_common * dz;
                return;
            }

            double s = node->half_side * 2.0;
            double rx = node->com.x - pos[i].x, ry = node->com.y - pos[i].y, rz = node->com.z - pos[i].z;
            double d = std::sqrt(rx*rx + ry*ry + rz*rz + eps2);
            if (s / d < theta) {
                double inv_d = 1.0 / d;
                double inv_d3 = inv_d * inv_d * inv_d;
                double factor = G * node->mass * inv_d3;

                acc.x += factor * rx, acc.y += factor * ry, acc.z += factor * rz;
                return;
            }

            else {
                for (const auto& child : node->children) {
                    if (!child || child->mass == 0.0) continue;

                    compute_node_forces(child.get(), i, acc);
                }
            }
        }

        void traverse_for_collisions(Node* node, int i) {
            if (!node) return;

            const vec3& p_i = pos[i];
            double r_i = radii[i];

            double dist = point_to_node_distance(node, p_i);

            if (dist > (r_i + node->max_radius)) return;

            // case one: Leat node contain one body
            if (node->children[0] == nullptr) {
                int j = node->body_index;
                
                if (j == -1 || i == j) return;
                else {
                    double dx = p_i.x - pos[j].x, dy = p_i.y - pos[j].y, dz = p_i.z - pos[j].z;
                    
                    double distance = std::sqrt(dx*dx + dy*dy + dz*dz);
                    double radius_sum = r_i + radii[j];

                    if (distance < radius_sum) {
                        if (i < j) collision_pairs.emplace_back(i, j);
                    }
                }
            }

            // case two: internal nodes
            else {
                for (const auto& child : node->children) {
                    traverse_for_collisions(child.get(), i);
                }
            }
        }

        double compute_node_potential(Node* node, int i) {
            // base case: Empty Node
            if (node->mass == 0.0) return 0.0;

            double phi_i = 0.0;

            // case one: Leaf Node
            if (node->children[0] == nullptr && node->body_index != -1) {
                if (node->body_index == i) return 0.0;

                else {
                    int j = node->body_index;
                    double dx = pos[j].x - pos[i].x, dy = pos[j].y - pos[i].y, dz = pos[j].z - pos[i].z;
                    double inv_dist = 1.0 / std::sqrt(dx*dx + dy*dy + dz*dz + eps2);
                    return -G * masses[j] * inv_dist;
                }
            }

            // case two: Internal Node
            double s = node->half_side * 2.0;
            double rx = node->com.x - pos[i].x, ry = node->com.y - pos[i].y, rz = node->com.z - pos[i].z;
            double d = std::sqrt(rx*rx + ry*ry + rz*rz + eps2);

            if (s / d < theta) {
                double inv_d = 1.0 / d;
                return -G * node->mass * inv_d;
            }

            else {
                for (const auto& child : node->children) {
                    if (!child || child->mass == 0.0) continue;
                    
                    phi_i += compute_node_potential(child.get(), i);
                }
                return phi_i;
            }
        }

};