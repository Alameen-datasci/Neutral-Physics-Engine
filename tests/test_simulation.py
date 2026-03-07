"""
Unit tests for the Simulation class in the neutral_physics_engine package.

This module validates the core simulation mechanics and physical properties of the engine.
Key functionalities tested include:
* **State Management:** Packing and unpacking state arrays for the numerical integrators.
* **Physical Quantities:** Accurate computation of kinetic energy, potential energy, linear momentum, and angular momentum.
* **Collision Handling:** Resolving elastic collisions using restitution coefficients and preventing object overlap.
* **Time Management:** Properly limiting integration time steps (`dt`) to strictly match the requested simulation duration.
"""

import numpy as np
import pytest
from neutral_physics_engine.simulation import Simulation
from neutral_physics_engine.forces import forces
from neutral_physics_engine.integrators import euler
from neutral_physics_engine.body import Body


def make_sim():
    """
    Create a simulation with two bodies: the sun and the earth.
    """
    sun = Body(mass=1.989e30, pos=[0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0], radius=6.96e8)
    earth = Body(
        mass=5.972e24, pos=[1.49e11, 0.0, 0.0], vel=[0.0, 29780.0, 0.0], radius=6.371e6
    )
    return Simulation(bodies=[sun, earth], force_fn=forces, integrator=euler, dt=1.0)


def test_pack_unpack_state():
    """
    Test the packing and unpacking of the simulation state to ensure that the positions and velocities of the bodies are correctly converted to and from a flat array format.
    This is important for the integrator to work correctly, as it relies on the state being represented as a single array for the numerical integration process.
    The test checks that the initial packing of the state matches the expected format and that unpacking a new state correctly updates the positions and velocities of the bodies in the simulation.
    This ensures that the internal state management of the Simulation class is functioning as intended.
    """
    # Create a simulation instance with known initial conditions
    sim = make_sim()
    # Test packing
    state = sim._pack_state()
    # The expected state is a flat array containing the positions and velocities of the sun and the earth in the order: [sun_pos, earth_pos, sun_vel, earth_vel]
    expected_state = np.array(
        [0.0, 0.0, 0.0, 1.49e11, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 29780.0, 0.0]
    )
    np.testing.assert_array_equal(state, expected_state)

    # Test unpacking
    random_new_state = np.array(
        [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0]
    )
    sim._unpack_state(random_new_state)

    # Check that the bodies' positions and velocities were updated correctly
    np.testing.assert_array_equal(sim.bodies[0].pos, [1.0, 2.0, 3.0])
    np.testing.assert_array_equal(sim.bodies[0].vel, [7.0, 8.0, 9.0])
    np.testing.assert_array_equal(sim.bodies[1].pos, [4.0, 5.0, 6.0])
    np.testing.assert_array_equal(sim.bodies[1].vel, [10.0, 11.0, 12.0])


def test_compute_energy():
    """
    Test the computation of kinetic and potential energy for the simulation bodies.
    This test verifies that the energy calculations are accurate based on the initial conditions of the bodies in the simulation.
    The kinetic energy is calculated using the formula KE = 0.5 * m * v^2 for each body, and the potential energy is calculated using
    the gravitational potential energy formula PE = -G * m1 * m2 / r for each pair of bodies, where G is the gravitational constant, m1 and m2 are the masses of the bodies, and r is the distance between them.
    The test compares the computed kinetic and potential energy against expected values that are derived from the known initial conditions of the sun and the earth in the simulation.
    This ensures that the energy calculations are implemented correctly and that the simulation's energy conservation properties can be accurately analyzed in future tests.
    """
    # Create a simulation instance with known initial conditions
    sim = make_sim()
    # Initial energy
    KE, PE = sim._compute_energy()

    # Pure Mathematical calculation of kinetic energy and potential energy
    expected_KE = 2.6481293224e33
    expected_PE = -5.320764502308725e33
    expected_E = -2.672635179908725e33

    np.testing.assert_allclose(KE, expected_KE, rtol=1e-12)
    np.testing.assert_allclose(PE, expected_PE, rtol=1e-12)
    np.testing.assert_allclose(KE + PE, expected_E, rtol=1e-12)


def test_compute_momenta():
    """
    Test the computation of linear and angular momenta for the simulation bodies.
    This test verifies that the momenta calculations are accurate based on the initial conditions of the bodies in the simulation.
    The linear momentum is calculated using the formula P = m * v for each body, and the angular momentum is calculated using
    the formula L = r x (m * v) for each body, where r is the position vector of the body relative to a chosen origin (in this case, the center of mass of the system).
    The test compares the computed linear and angular momenta against expected values that are derived from the known initial conditions of the sun and the earth in the simulation.
    This ensures that the momenta calculations are implemented correctly and that the simulation's momentum conservation properties can be accurately analyzed in future tests.
    """
    # Create a simulation instance with known initial conditions
    sim = make_sim()
    # Initial momenta
    P, L = sim._compute_momenta()

    # Pure Mathematical calculation of linear momentum and angular momentum
    expected_P = np.array([0.0, 1.7784616e29, 0.0])
    expected_L = np.array([0.0, 0.0, 2.64989982763913e40])

    np.testing.assert_allclose(P, expected_P, rtol=1e-12)
    np.testing.assert_allclose(L, expected_L, rtol=1e-12)


def test_resolve_collisions():
    """
    Test the collision resolution between two bodies in the simulation.
    This test creates a scenario where two bodies are on a collision course and checks that after the collision is resolved, the velocities of the bodies are updated correctly according to the conservation of momentum and the specified restitution coefficient.
    The test also verifies that the distance between the bodies after the collision is equal to the sum of their radii, ensuring that they are just touching and not overlapping, which is a key aspect of the collision resolution process.
    This ensures that the collision handling in the Simulation class is functioning correctly and that the physical interactions between bodies are accurately modeled during collisions.
    """
    # Create a simulation with two bodies on a collision course
    b1 = Body(mass=1.0, pos=[0.0, 0.0, 0.0], vel=[1.0, 0.0, 0.0], radius=1.0)
    b2 = Body(mass=1.0, pos=[1.5, 0.0, 0.0], vel=[-1.0, 0.0, 0.0], radius=1.0)
    sim = Simulation(
        bodies=[b1, b2], force_fn=forces, integrator=euler, dt=0.1, restitution=1.0
    )
    sim._handle_collisions()

    np.testing.assert_allclose(b1.vel, np.array([-1.0, 0.0, 0.0]))
    np.testing.assert_allclose(b2.vel, np.array([1.0, 0.0, 0.0]))
    # After the collision, the bodies should be just touching, so the distance between their centers should be equal to the sum of their radii (1.0 + 1.0 = 2.0)
    dist = np.linalg.norm(b1.pos - b2.pos)
    assert dist == 2.0


def test_collision_moving_apart():
    """
    Test the collision resolution for two bodies that are moving apart after a collision.
    This test creates a scenario where two bodies have already collided and are moving away from each other, and checks that the collision resolution does not alter their velocities since they are no longer in contact.
    """
    # Create a simulation with two bodies moving apart after a collision
    b1 = Body(mass=1.0, pos=[0.0, 0.0, 0.0], vel=[-1.0, 0.0, 0.0], radius=1.0)
    b2 = Body(mass=1.0, pos=[1.5, 0.0, 0.0], vel=[1.0, 0.0, 0.0], radius=1.0)
    sim = Simulation(
        bodies=[b1, b2], force_fn=forces, integrator=euler, dt=0.1, restitution=1.0
    )

    sim._handle_collisions()

    np.testing.assert_allclose(b1.vel, np.array([-1.0, 0.0, 0.0]))
    np.testing.assert_allclose(b2.vel, np.array([1.0, 0.0, 0.0]))
    # The bodies are moving apart, so the distance between their centers should be greater than the sum of their radii (1.0 + 1.0 = 2.0)
    dist = np.linalg.norm(b1.pos - b2.pos)
    assert dist == 1.5


def test_run_limits_dt():
    """
    Test that the run method correctly limits the time step to ensure that the total simulation time is not exceeded.
    This test creates a simulation with a specified total time and time step, and checks that after running the simulation, the final time does not exceed the total time, even if the time step would cause it to do so.
    This is important for ensuring that the simulation runs for the intended duration and that the time step is adjusted appropriately in the final iteration to prevent overshooting the total simulation time.
    """
    # Create a simulation with a single body and a time step that would exceed the total time
    b1 = Body(mass=1.0, pos=[0.0, 0.0, 0.0], vel=[0.0, 0.0, 0.0], radius=1.0)
    sim = Simulation([b1], euler, forces, dt=0.1)
    # Run the simulation for a total time of 0.25 seconds, which would require 3 steps of 0.1 seconds each, but the last step should be limited to 0.05 seconds to not exceed the total time
    sim.run(0.25)

    assert sim.t == 0.25
