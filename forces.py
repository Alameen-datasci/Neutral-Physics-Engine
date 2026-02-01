import numpy as np

def forces(bodies):
    G = 6.67430e-11
    eps = 1e-10
    accs = []

    for i, bi in enumerate(bodies):
        ai = np.zeros_like(bi.pos)

        # -----RESTING CONTACT FIX-----
        if not (bi.pos[-1] == 0 and bi.vel[-1] == 0):
            ai[-1] = -9.8
        # -----------------------------
        
        for j, bj in enumerate(bodies):
            if i == j:
                continue
            r = bj.pos - bi.pos
            ai += (G * bj.mass / (np.linalg.norm(r)**3 + eps)) * r

        accs.append(ai)

    return accs