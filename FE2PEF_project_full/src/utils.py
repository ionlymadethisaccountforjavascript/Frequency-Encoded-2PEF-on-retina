import numpy as np

def create_synthetic_fluorophore_map(nx=128, ny=128, types=2, spots_per_type=6, seed=0):
    rng = np.random.RandomState(seed)
    maps = np.zeros((types, ny, nx))
    for t in range(types):
        for s in range(spots_per_type):
            cx = rng.randint(int(0.1*nx), int(0.9*nx))
            cy = rng.randint(int(0.1*ny), int(0.9*ny))
            sigma = rng.uniform(3, 12)
            x = np.arange(nx)
            y = np.arange(ny)
            X, Y = np.meshgrid(x, y, indexing='xy')
            spot = np.exp(-((X-cx)**2 + (Y-cy)**2)/(2*sigma**2))
            amp = rng.uniform(0.5, 1.5)
            maps[t] += amp*spot
    for t in range(types):
        if maps[t].sum()>0:
            maps[t] /= maps[t].max()
    return maps
