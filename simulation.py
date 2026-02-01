import numpy as np
import matplotlib.pyplot as plt

class Simulation:
    def __init__(self, bodies, integrator, force_fn, dt, restitution=0.8):
        self.bodies = bodies
        self.integrator = integrator
        self.force_fn = force_fn
        self.dt = dt
        self.t = 0
        self.r = restitution
        self.traj = [[] for _ in bodies]
        self.vels = [[] for _ in bodies]
        self.time = []

    def step(self):
        self.integrator(self.bodies, self.force_fn, self.dt, self.t)

        for i, b in enumerate(self.bodies):
            if b.pos[-1] <= 0:
                b.pos[-1] = 0
                if b.vel[-1] < 0:
                    b.vel[-1] = -self.r * b.vel[-1]

            self.traj[i].append(b.pos.copy())
            self.vels[i].append(b.vel.copy())

        self.t += self.dt
        self.time.append(self.t)

    def run(self, T):
        for _ in range(int(T / self.dt)):
            self.step()

    def plot(self):
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))
        for i in range(len(self.bodies)):
            h = np.array(self.traj[i])[:, -1]
            v = np.linalg.norm(self.vels[i], axis=1)
            ax[0].plot(self.time, h)
            ax[0].set_xlabel('Time (s)')
            ax[0].set_ylabel('Height (m)')
            ax[1].plot(self.time, v)
            ax[1].set_xlabel('Time (s)')
            ax[1].set_ylabel('Speed (m/s)')
        plt.show()