# ============================================================================
# IMPORTS - EVERYTHING YOU NEED
# ============================================================================
import os
os.environ["DDE_BACKEND"] = "tensorflow"
import deepxde as dde
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os
import tensorflow as tf
import math
print("DeepXDE backend:", dde.backend.backend_name)
# Set TensorFlow backend
dde.config.set_default_float("float32")
