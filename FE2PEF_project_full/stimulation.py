'''
    # stimulation.py (partial) - user provided chunks assembled here
    # This file is a partial copy of retinasim/stimulation.py you pasted; it's incomplete.
    import numpy as np
    cp = np
    arr = np.asarray
    import math, os, sys, random, json, shutil, time
    from matplotlib import pyplot as plt
    from scipy import ndimage
    import psutil
    from multiprocessing import Process, Queue
    from retinasim import geometry
    from retinasim.lsystemlib.utilities import *
    from retinasim.lsystemlib.embedding import EmbedVessels
    from retinasim.lsystemlib.tree import Tree
    from retinasim.lsystemlib.growth_process import GrowthProcess, Params

    num_cpus = psutil.cpu_count(logical=False)
    sphere_centre = arr([0.,-20000.,0.])
    np.random.seed()

    class Simulation(object):
        def __init__(self, prefix='', path=None, store_graph=False, store_mesh=False, store_density=False,
                     store_vegf=False, store_vessel_grid=False, store_every=1, write_longitudinal=False,
                     verbose=2, to_shared_memory=False, from_shared_memory=False, n=10000, logging=False,
                     create_terminals=True, domain_size=[1000.,1000.,1000.], domain=[None,None,None],
                     domain_type='cuboid', ignore_domain=False, embedding_resolution=None, embedding_dim=None,
                     embed=True, embedding_offset=arr([0.,0.,0.]), tree=None, initialise_folders=True,
                     n_sampling_directions=21, planar=False, match_embedding_dims=False, max_cycles=50, **kwargs):
            self.verbose = verbose
            self.tree = tree
            self.set_domain(domain_size,domain=domain,domain_type=domain_type)
            self.current_cycle = 0
            self.params = Params()
            self.max_cycles = max_cycles
            # ... many methods omitted in this demo file
            print('Simulation partial loaded. This is a truncated copy; paste the rest of stimulation.py into this file.')
    if __name__ == '__main__':
        print('stimulation.py partial demo. Replace with the full file from retinasim for full functionality.')
'''