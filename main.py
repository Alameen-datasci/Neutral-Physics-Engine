"""
main.py

This script sets up and runs a physics simulation of a meteor falling towards Earth under gravity. It initializes the Earth and a meteor, configures the simulation parameters, and executes the simulation while plotting the results.
"""
from body import Body, Earth
from forces import forces
from integrators import euler_step, rk4_body_step, velocity_verlet_step
from simulation import Simulation
import numpy as np

# --- 1. Create Bodies ---

# Initialize Earth (Radius ~6,371,008 m)
earth = Earth()

# Initialize a Meteor
# Position: 500 meters strictly above Earth's surface
# Velocity: 0 (Stationary release)
meteor_pos = [Earth.RADIUS + 500.0, 0, 0]
meteor_vel = [0, 0, 0]

meteor = Body(
    mass=2000.0,
    pos=meteor_pos,
    vel=meteor_vel,
    radius=5.0
)

# Add both to the system
bodies = [earth, meteor]

# --- 2. Setup Simulation ---

# Time step (dt) set to 0.5 seconds for collision precision without stress
# Total time (T) set to 15 seconds (enough for a 500m drop)
dt = 0.5
T = 15.0

sim = Simulation(
    bodies=bodies,
    integrator=velocity_verlet_step,  # You can switch to rk4_body_step or velocity_verlet_step for better accuracy
    force_fn=forces,
    dt=dt
)

# --- 3. Run Simulation ---
sim.run(T)
# --- 4. Plot Simulation ---
sim.plot()

# The simulation results are now stored in sim.traj, sim.vels, etc.
# You can inspect them manually in your debugger or interactive shell.