"""
Unit tests for the Simulation orchestrator.
"""
import pytest
import numpy as np
from neutral_physics_engine.simulation import Simulation
from neutral_physics_engine.body import Body
from neutral_physics_engine.integrators import euler
from neutral_physics_engine.gravity_field import GravityField

def create_dummy_system():
    """Helper to generate a simple 2-body system for testing."""
    b1 = Body(mass=1.0, pos=[0, 0, 0], vel=[1, 0, 0], radius=0.5)
    b2 = Body(mass=1.0, pos=[10, 0, 0], vel=[-1, 0, 0], radius=0.5)
    return [b1, b2], GravityField()

def test_simulation_initialization():
    """Test that the simulation initializes and correctly maps the integrator."""
    bodies, field = create_dummy_system()
    sim = Simulation(bodies, euler, field, dt=0.1, time_stepping="fixed", enable_collisions=False)
    
    assert sim.t == 0.0
    assert sim.dt == 0.1
    assert sim.time_stepping == "fixed"
    assert not sim.enable_collisions

def test_simulation_pack_unpack_state():
    """Test the phase-space vector flattening and un-flattening."""
    bodies, field = create_dummy_system()
    sim = Simulation(bodies, euler, field, dt=0.1, time_stepping="fixed", enable_collisions=False)
    
    state = sim._pack_state()
    # 2 bodies * 3 dims * 2 (pos + vel) = 12 elements
    assert state.shape == (12,)
    np.testing.assert_array_equal(state[:3], [0, 0, 0])  # Body 1 pos
    np.testing.assert_array_equal(state[6:9], [1, 0, 0]) # Body 1 vel
    
    # Modify state and unpack
    new_state = np.zeros(12)
    new_state[0] = 5.0 # Change Body 1 X-position
    sim._unpack_state(new_state)
    
    # Check that the Body object itself was updated
    assert sim.bodies[0].pos[0] == 5.0

def test_simulation_run_exact_time():
    """
    Test that sim.run(T) truncates the final time step to end exactly at T, 
    preventing over-integration.
    """
    bodies, field = create_dummy_system()
    # Use dt = 0.3. Running to 1.0 should take steps: 0.3, 0.3, 0.3, and exactly 0.1
    sim = Simulation(bodies, euler, field, dt=0.3, time_stepping="fixed", enable_collisions=False)
    
    sim.run(T=1.0)
    
    # Time should be exactly 1.0
    assert np.isclose(sim.t, 1.0)
    # The original dt (0.3) should have been restored after the truncated final step
    assert sim.dt == 0.3