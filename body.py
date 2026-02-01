import numpy as np

class Body:
    def __init__(self, mass, pos, vel):
        self.mass = mass
        self.pos = np.array(pos, dtype=float)
        self.vel = np.array(vel, dtype=float)