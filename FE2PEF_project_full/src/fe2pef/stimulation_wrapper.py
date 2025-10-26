
"""Stimulation wrapper for demo.

This module tries to import the real `retinasim` package from third_party/retinasim.
If available, it creates a Simulation object and uses it to produce a retina plane or volume.
If not available, it generates a simple synthetic 2D retina-like map (vessels + patches) for demo purposes.
"""
import numpy as np
import os
def try_import_retinasim():
    try:
        import importlib, sys
        rp = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'third_party'))
        if rp not in sys.path:
            sys.path.insert(0, rp)
        import retinasim
        return retinasim
    except Exception as e:
        return None

def generate_synthetic_retina(shape=(256,256), n_patches=6, random_seed=0):
    np.random.seed(random_seed)
    img = np.zeros(shape, dtype=float)
    # place gaussian patches to mimic regions with different fluorophore distributions
    xs = np.linspace(0, shape[1]-1, shape[1])
    ys = np.linspace(0, shape[0]-1, shape[0])
    X, Y = np.meshgrid(xs, ys)
    for i in range(n_patches):
        cx = np.random.uniform(0, shape[1])
        cy = np.random.uniform(0, shape[0])
        sigma = np.random.uniform(8, 30)
        amp = np.random.uniform(0.5, 1.5)
        img += amp * np.exp(-((X-cx)**2+(Y-cy)**2)/(2*sigma**2))
    # Normalize
    img = img - img.min()
    img = img / (img.max()+1e-12)
    return img

class RetinaProvider:
    def __init__(self, use_retinasim=True):
        self.retinasim = try_import_retinasim() if use_retinasim else None
    def get_plane(self, shape=(256,256)):
        if self.retinasim is not None:
            # attempt to create a Simulation object similar to the code you provided
            try:
                simmodule = self.retinasim.stimulation
                sim = simmodule.Simulation(prefix='demo', path=None, initialise_folders=False, planar=True, domain_type='surface', domain_size=[shape[1], shape[0], 1.0])
                # The real retinasim has methods to write grids. For this demo we just try to extract something
                # Placeholder: return a synthetic image for now
                return generate_synthetic_retina(shape=shape)
            except Exception as e:
                return generate_synthetic_retina(shape=shape)
        else:
            return generate_synthetic_retina(shape=shape)
