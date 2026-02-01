"""
neutral-physics-engine

Entry point of the project.
Creates bodies, runs simulations using different integrators,
and visualizes results.
"""

from body import Body
from forces import forces
from integrators import euler_step, rk4_body_step
from simulation import Simulation

b1 = Body(1.0, [5, 5], [0, 10])
# b2 = Body(2.0, [2, 10], [0, 0])

sim_euler = Simulation(
    bodies=[Body(1.0, [5, 5], [0, 10])],
    integrator=euler_step,
    force_fn=forces,
    dt=0.01
)

sim_rk4 = Simulation(
    bodies=[Body(1.0, [5, 5], [0, 10])],
    integrator=rk4_body_step,
    force_fn=forces,
    dt=0.01
)

# sim_euler.run(5)
sim_rk4.run(5)

# sim_euler.plot()
sim_rk4.plot()