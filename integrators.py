"""
Numerical integration methods.

Implements Euler and Runge-Kutta (RK4) time-stepping
for advancing the system state.
"""

import numpy as np

def euler_step(bodies, force_fn, dt, t):
    accs = force_fn(bodies)
    for i, b in enumerate(bodies):
        b.vel += accs[i] * dt
        b.pos += b.vel * dt

def rk4_step(fun, dt, t, x):
    k1 = fun(t, x)
    k2 = fun(t + dt/2, x + dt/2 * k1)
    k3 = fun(t + dt/2, x + dt/2 * k2)
    k4 = fun(t + dt, x + dt * k3)
    return x + dt/6 * (k1 + 2*k2 + 2*k3 + k4)

def derivative(bodies, force_fn):
    n = len(bodies)
    dim = bodies[0].pos.size
    masses = [b.mass for b in bodies]

    def f(t, x):
        pos = x[:n*dim].reshape(n, dim)
        vel = x[n*dim:].reshape(n, dim)

        temp = []
        for i in range(n):
            temp.append(type(bodies[i])(masses[i], pos[i], vel[i]))

        accs = force_fn(temp)

        dx = np.zeros_like(x)
        dx[:n*dim] = vel.flatten()
        dx[n*dim:] = np.array(accs).flatten()
        return dx

    return f

def rk4_body_step(bodies, force_fn, dt, t):
    n = len(bodies)
    dim = bodies[0].pos.size

    x0 = np.concatenate(
        [np.array([b.pos for b in bodies]).flatten(),
         np.array([b.vel for b in bodies]).flatten()]
    )

    f = derivative(bodies, force_fn)
    x1 = rk4_step(f, dt, t, x0)

    pos = x1[:n*dim].reshape(n, dim)
    vel = x1[n*dim:].reshape(n, dim)

    for i in range(n):
        bodies[i].pos = pos[i]
        bodies[i].vel = vel[i]