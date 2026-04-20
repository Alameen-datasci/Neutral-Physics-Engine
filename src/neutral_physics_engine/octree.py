"""
octree.py

This module implements the Barnes-Hut octree data structure for high-performance
N-body gravitational simulations. It provides an O(N log N) approximation algorithm
for computing accelerations and gravitational potentials, making it suitable for
large-scale astrophysical, planetary, or spacecraft dynamics simulations (NASA/ESA-grade
professional physics engines).

The Octree class builds a hierarchical 3D spatial partitioning tree where each node
stores the total mass and center-of-mass of all bodies it contains. This enables
the Barnes-Hut opening-angle criterion (controlled by `theta`) to decide whether to
approximate a distant cluster of bodies as a single point mass or to recurse into
children for higher accuracy.

Key features:
- Efficient tree construction and insertion with automatic subdivision
- Recursive force/acceleration evaluation with Barnes-Hut approximation
- Per-body gravitational potential computation (for total energy diagnostics)
- Optional collision detection using per-body radii and node-level max-radius
  bounding-volume culling (perfect for rigid-body or particle simulations)
- Validation methods (e.g., compare_with_direct) for accuracy benchmarking

The Node class is an internal building block representing either a leaf (single body)
or an internal node (subdivision). All methods are designed for repeated use in
time-stepping loops with minimal overhead.

Constants:
    G   : Gravitational constant (SI units)
    EPS : Softening parameter to prevent singularities
"""

import numpy as np


G = 6.67430e-11  # Gravitational constant in m³ kg⁻¹ s⁻² (CODATA 2018 value)
EPS = 1e-10  # Small value used to avoid division-by-zero in distance calculations


# ========================== Node ==========================
class Node:
    """
    Represents a single node in the Barnes-Hut octree.

    Each node stores spatial information (center and half-side length), aggregate
    physics quantities (total mass, center-of-mass, maximum radius when radii are
    supplied), and either a single body index (leaf node) or a list of 8 child nodes
    (internal node).

    This class is used internally by Octree and is not intended for direct
    instantiation outside the tree-building process.
    """

    def __init__(self, center: list[float] | np.ndarray, half_side: np.float64):
        """
        Initialize a new octree node.

        Parameters:
        -----------
        center : list[float] | np.ndarray
            3D coordinates of the node center (shape (3,))
        half_side : np.float64
            Half the side length of the cubic node (must be positive)

        Raises:
        -------
        ValueError
            If center is not a 3D vector or half_side is non-positive.
        """
        self.center = np.asarray(center, dtype=np.float64)

        if self.center.shape != (3,):
            raise ValueError("Node center must be a 3D vector of shape (3,)")

        if half_side <= 0:
            raise ValueError("Node half_side must be positive")

        self.half_side = half_side
        self.body_index = None  # index of the body if this is a leaf, else None
        self.children = None  # list of 8 child Node objects or None
        self.mass = 0.0  # total mass of all bodies in this subtree
        self.com = np.zeros(3)  # center-of-mass of all bodies in this subtree
        self.max_radius = (
            0.0  # maximum radius of any body in this subtree (when radii supplied)
        )


# ========================== Octree ==========================
class Octree:
    """
    Barnes-Hut octree for efficient N-body gravity and collision queries.

    This is the core spatial data structure used by GravityField. It supports:
    - Fast approximate force calculations via the Barnes-Hut criterion
    - Gravitational potential evaluation
    - Optional collision-pair detection when body radii are provided

    The tree is rebuilt from scratch for each new set of positions (typical in
    simulation time steps). Caching of the tree structure happens in the calling
    GravityField class.
    """

    def __init__(
        self,
        masses: list[float] | np.ndarray,
        pos: list[float] | np.ndarray,
        radii: list[float] | np.ndarray | None = None,
        theta: np.float64 = 0.5,
    ):
        """
        Initialize the octree with the current system state.

        Parameters:
        -----------
        masses : list[float] | np.ndarray
            List of body masses (length N)
        pos : list[float] | np.ndarray
            List of body positions (N arrays/vectors of shape (3,))
        radii : list[float] | np.ndarray | None, optional
            List of body radii for collision detection (length N). If None,
            collision methods will raise an error.
        theta : np.float64, optional
            Barnes-Hut opening angle parameter (default 0.5). Controls the
            accuracy/performance trade-off. Lower values = higher accuracy.
        """
        self.masses = masses
        self.pos = pos
        self.root = None
        self.theta = theta
        self.radii = radii
        self.collision_pairs = []

    # --------------------- Building the tree ---------------------
    def build(self) -> None:
        """
        Build the entire octree from the current set of bodies.

        Computes the bounding box of all particles, creates the root node with
        appropriate size, then inserts every body into the tree.

        Raises:
        -------
        ValueError
            If no bodies are present.
        """
        if self.pos.shape[0] == 0:
            raise ValueError(
                "Cannot build octree: positions array must contain at least one body"
            )

        x0, y0, z0 = self.pos[0]

        xmin = xmax = x0
        ymin = ymax = y0
        zmin = zmax = z0

        for p in self.pos[1:]:
            x, y, z = p

            xmin = min(xmin, x)
            xmax = max(xmax, x)

            ymin = min(ymin, y)
            ymax = max(ymax, y)

            zmin = min(zmin, z)
            zmax = max(zmax, z)

        dx = xmax - xmin
        dy = ymax - ymin
        dz = zmax - zmin

        padding = 1e-5
        half_side = max(dx, dy, dz) / 2 + padding

        cx = (xmax + xmin) / 2
        cy = (ymax + ymin) / 2
        cz = (zmax + zmin) / 2

        self.root = Node([cx, cy, cz], half_side)

        for i in range(len(self.masses)):
            self.insert(i)

    # --------------------- Inserting bodies ---------------------
    def insert(self, i) -> None:
        """
        Insert body i into the octree (public wrapper).

        Parameters:
        -----------
        i : int
            Index of the body to insert
        """
        if self.root is None:
            raise ValueError(
                "Cannot insert body: octree must be built first (call build() before insert())"
            )
        self._insert(self.root, i)

    def _insert(self, node, i) -> None:
        """
        Recursively insert body i into the subtree rooted at `node`.

        Updates mass, center-of-mass, and max_radius on every ancestor node.
        Handles leaf-to-internal promotion and subdivision automatically.
        """
        old_mass = node.mass
        new_mass = old_mass + self.masses[i]
        if old_mass == 0:
            node.com = self.pos[i].copy()
        else:
            node.com = (old_mass * node.com + self.masses[i] * self.pos[i]) / new_mass
        node.mass = new_mass

        if self.radii is not None:
            node.max_radius = max(node.max_radius, self.radii[i])

        # case 1 - empty leaf
        if node.body_index is None and node.children is None:
            node.body_index = i
            return
        # case 2 - leaf node with one body
        elif node.body_index is not None and node.children is None:
            old_body_index = node.body_index
            node.body_index = None

            node.children = self.subdivide(node)

            self._insert(
                self.choose_child(node, self.pos[old_body_index]), old_body_index
            )

            self._insert(self.choose_child(node, self.pos[i]), i)

            return
        # case 3 - internal node
        else:
            child = self.choose_child(node, self.pos[i])
            self._insert(child, i)

    # --------------------- Subdivision and child selection ---------------------
    def subdivide(self, node) -> list:
        """
        Create and return the eight child nodes for a given parent node.

        Parameters:
        -----------
        node : Node
            Parent node to subdivide

        Returns:
        --------
        list[Node]
            List of 8 new child Node objects
        """
        child_half = node.half_side / 2
        cx, cy, cz = node.center
        children = []
        for dx in (-child_half, child_half):
            for dy in (-child_half, child_half):
                for dz in (-child_half, child_half):
                    center = np.array([cx + dx, cy + dy, cz + dz], dtype=np.float64)
                    children.append(Node(center, child_half))
        return children

    def choose_child(self, node: Node, pos: np.ndarray) -> Node:
        """
        Return the appropriate child node for a given position.

        Parameters:
        -----------
        node : Node
            Parent node
        pos : np.ndarray
            Position vector (shape (3,))

        Returns:
        --------
        Node
            The child node corresponding to the octant containing `pos`
        """
        index = self.get_octant_index(node, pos)
        return node.children[index]

    def get_octant_index(self, node: Node, pos: np.ndarray) -> int:
        """
        Compute the octant index (0-7) for a position relative to a node center.

        Uses the standard binary indexing: x*4 + y*2 + z*1 where each component
        is 0 for negative and 1 for non-negative.

        Parameters:
        -----------
        node : Node
            Reference node
        pos : np.ndarray
            Position to classify

        Returns:
        --------
        int
            Octant index (0 ≤ index ≤ 7)
        """
        diff = np.subtract(pos, node.center)
        ix = 1 if diff[0] >= 0 else 0
        iy = 1 if diff[1] >= 0 else 0
        iz = 1 if diff[2] >= 0 else 0
        return ix * 4 + iy * 2 + iz

    # --------------------- Force calculation ---------------------
    def compute_acceleration(self, i: int):
        """
        Compute the gravitational acceleration on body i using the Barnes-Hut
        approximation.

        Public interface used by GravityField.__call__.

        Parameters:
        -----------
        i : int
            Index of the body

        Returns:
        --------
        np.ndarray
            Acceleration vector (shape (3,)) acting on body i
        """
        return self._compute_node_forces(self.root, i)

    def _compute_node_forces(self, node: Node, i: int):
        """
        Recursively compute the acceleration on body i due to the subtree rooted
        at `node`, applying the Barnes-Hut opening-angle criterion.
        """
        if node.mass < EPS:
            return np.zeros(3)

        a_i = np.zeros(3)

        if node.children is None and node.body_index == i:
            return np.zeros(3)

        if node.children is None and node.body_index != i:
            j = node.body_index
            disp = np.subtract(self.pos[j], self.pos[i])
            dist = max(np.linalg.norm(disp), EPS)
            f_common = G / (dist**3)
            return f_common * self.masses[j] * disp

        if node.children is not None:
            s = node.half_side * 2
            r = np.subtract(node.com, self.pos[i])
            d = np.linalg.norm(r)
            if d < EPS:
                d = EPS
            if s / d < self.theta:
                grav_coeff = G / (d**3)
                return grav_coeff * node.mass * r
            else:
                for child in node.children:
                    if child.mass == 0:
                        continue
                    a_i += self._compute_node_forces(child, i)
                return a_i

    def compare_with_direct(self, i: int) -> tuple:
        """
        Compare the tree-approximated acceleration with the exact direct summation
        for validation and accuracy benchmarking.

        Useful during development or when tuning the `theta` parameter.

        Parameters:
        -----------
        i : int
            Body index

        Returns:
        --------
        tuple[np.ndarray, np.ndarray, float]
            (tree_acc, direct_acc, absolute_difference_norm)
        """
        tree_acc = self.compute_acceleration(i)
        direct_acc = np.zeros(3)
        for j in range(len(self.masses)):
            if i == j:
                continue
            dr = np.subtract(self.pos[j], self.pos[i])
            r = max(np.linalg.norm(dr), EPS)
            direct_acc += G * self.masses[j] / r**3 * dr
        return tree_acc, direct_acc, np.linalg.norm(tree_acc - direct_acc)

    # --------------------- Collision detection ---------------------
    def point_to_node_distance(self, node: Node, position: np.ndarray) -> float:
        """
        Compute the shortest distance from a point to the surface of a node's cube.

        Used for efficient early-out culling in collision detection.

        Returns:
        --------
        float
            Distance (0 if point is inside the node)
        """
        cx, cy, cz = node.center
        h = node.half_side
        x, y, z = position

        if x < (cx - h):
            dx = (cx - h) - x
        elif x > (cx + h):
            dx = x - (cx + h)
        else:
            dx = 0

        if y < (cy - h):
            dy = (cy - h) - y
        elif y > (cy + h):
            dy = y - (cy + h)
        else:
            dy = 0

        if z < (cz - h):
            dz = (cz - h) - z
        elif z > (cz + h):
            dz = z - (cz + h)
        else:
            dz = 0

        return np.sqrt(dx**2 + dy**2 + dz**2)

    def find_collisions(self):
        """
        Find all pairs of bodies whose spheres intersect.

        Uses the octree and per-node max_radius for O(N log N) performance with
        early culling. Only works when radii were supplied at construction.

        Returns:
        --------
        list[tuple[int, int]]
            List of colliding body index pairs (i < j)
        """
        if self.radii is None:
            raise ValueError("Radii required for collision detection")

        self.collision_pairs = []
        for i in range(len(self.masses)):
            self._traverse_for_collisions(self.root, i)
        return self.collision_pairs

    def _traverse_for_collisions(self, node: Node, i: int):
        """
        Recursively traverse the subtree for collisions involving body i.
        """
        if node is None:
            return

        p_i, r_i = self.pos[i], self.radii[i]

        dist = self.point_to_node_distance(node, p_i)

        if dist > (r_i + node.max_radius):
            return

        # case one: leaf node and contain a body
        if node.children is None:
            j = node.body_index
            if j is None or i == j:
                return
            else:
                p_j, r_j = self.pos[j], self.radii[j]

                distance = np.linalg.norm(p_i - p_j)

                if distance < (r_i + r_j):
                    if i < j:
                        self.collision_pairs.append((i, j))

        # case two: internal nodes
        else:
            for child in node.children:
                self._traverse_for_collisions(child, i)

    # --------------------- Potential energy calculation ---------------------
    def compute_potential(self, i: int):
        """
        Compute the gravitational potential at body i due to all other bodies
        using the Barnes-Hut approximation.

        Public interface used by GravityField.compute_potential.

        Parameters:
        -----------
        i : int
            Body index

        Returns:
        --------
        float
            Gravitational potential at body i (excluding self-interaction)
        """
        return self._compute_node_potential(self.root, i)

    def _compute_node_potential(self, node: Node, i: int):
        """
        Recursively compute the potential at body i from the subtree rooted at
        `node`, applying the same Barnes-Hut criterion as force calculation.
        """
        # Base Case: Empty Node
        if node.mass < EPS:
            return 0.0

        phi_i = 0.0

        # Case One: Leaf Node
        if node.children is None and node.body_index is not None:
            if node.body_index == i:
                return 0.0
            else:
                j = node.body_index
                disp = np.subtract(self.pos[j], self.pos[i])
                dist = max(np.linalg.norm(disp), EPS)
                return -G * self.masses[j] / dist

        # Case Two: Internal Node
        if node.children is not None:
            s = node.half_side * 2
            r = np.subtract(node.com, self.pos[i])
            d = max(np.linalg.norm(r), EPS)

            if s / d < self.theta:
                return -G * node.mass / d
            else:
                for child in node.children:
                    if child.mass == 0:
                        continue
                    phi_i += self._compute_node_potential(child, i)
                return phi_i
