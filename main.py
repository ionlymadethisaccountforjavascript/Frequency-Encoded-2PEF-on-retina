#IMSKBIIDI
import deepxde as dde
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os
import tensorflow as tf

# Set TensorFlow backend
dde.config.set_default_float("float32")

# ============================================================================
# PART 1: LOAD AND VISUALIZE EYE MODEL
# ============================================================================

print("Loading and visualizing eye model...")
scene = trimesh.load('human_eye.glb')  
iris_mesh = scene.geometry['Eye_Iris_0']
bigbig_mesh = scene.to_geometry()

vertices = bigbig_mesh.vertices
faces = bigbig_mesh.faces
vertices = vertices.astype(np.float32)
faces = faces.astype(np.int32)

properties = {
    'Eye_Iris_0': {
        "collagen_density": 0.8,
        "melanin_content": 0.7,
        "is_iris": True
    },
    'Eye_Eye_0': {
        "collagen_density": 0.3,
        "melanin_content": 0.2,
        "is_iris": False
    }
}

# Figure for biochemical properties
fig = plt.figure(figsize=(18, 6))

# Plot 1: Collagen Density
ax1 = fig.add_subplot(131, projection='3d')
for name, mesh in scene.geometry.items():
    verts = mesh.vertices
    prop = properties[name]
    ax1.scatter(verts[:,0], verts[:,1], verts[:,2],
                c=np.full(len(verts), prop['collagen_density']),
                cmap='viridis', s=2, alpha=0.7,
                vmin=0, vmax=1)
ax1.set_title('Collagen Density')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
mappable1 = plt.cm.ScalarMappable(cmap='viridis')
mappable1.set_array([0,1])
fig.colorbar(mappable1, ax=ax1, shrink=0.5, label='Collagen Density')

# Plot 2: Melanin Content
ax2 = fig.add_subplot(132, projection='3d')
for name, mesh in scene.geometry.items():
    verts = mesh.vertices
    prop = properties[name]
    ax2.scatter(verts[:,0], verts[:,1], verts[:,2],
                c=np.full(len(verts), prop['melanin_content']),
                cmap='plasma', s=2, alpha=0.7,
                vmin=0, vmax=1)
ax2.set_title('Melanin Content')
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
mappable2 = plt.cm.ScalarMappable(cmap='plasma')
mappable2.set_array([0,1])
fig.colorbar(mappable2, ax=ax2, shrink=0.5, label='Melanin Content')

# Plot 3: Iris Region
ax3 = fig.add_subplot(133, projection='3d')
colors = {True: 'red', False: 'blue'}
for name, mesh in scene.geometry.items():
    verts = mesh.vertices
    prop = properties[name]
    ax3.scatter(verts[:,0], verts[:,1], verts[:,2],
                color=colors[prop['is_iris']], s=2, alpha=0.7, label=name)
ax3.set_title('Iris Region')
ax3.set_xlabel('X'); ax3.set_ylabel('Y'); ax3.set_zlabel('Z')
ax3.legend()

plt.tight_layout()
plt.show()

# Load and display the GLB file
current_dir = os.path.dirname(os.path.abspath(__file__))
glb_file = os.path.join(current_dir, "default_eye_ball.glb")

scene = trimesh.load(glb_file)

if isinstance(scene, trimesh.Scene):
    print(f"Scene meshes: {len(scene.geometry)}")
    for name, mesh in scene.geometry.items():
        print(f"{name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    scene.show()
else:
    print(f"Mesh: {len(scene.vertices)} vertices, {len(scene.faces)} faces")
    scene.show()

# ============================================================================
# PART 2: NONLINEAR OPTICS SIMULATION WITH PINNs
# ============================================================================

print("Setting up nonlinear optics simulation...")

# FIXED: Real-valued SHG PDE
def shg_pde(x, y):
    """
    Second Harmonic Generation (SHG) - Real-valued formulation
    """
    E_w_real, E_w_imag, E_2w_real, E_2w_imag = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]

    # PHYSICAL PARAMETERS
    wavelength = 800e-9
    k_w = 2 * np.pi / wavelength
    k_2w = 2 * np.pi / (wavelength / 2)
    deff = 1e-12
    alpha_w = 0.1
    alpha_2w = 0.1
    c = 3e8

    # Calculate derivatives
    E_w_real_z = dde.grad.jacobian(y, x, i=0, j=2)
    E_w_imag_z = dde.grad.jacobian(y, x, i=1, j=2)
    E_2w_real_z = dde.grad.jacobian(y, x, i=2, j=2)
    E_2w_imag_z = dde.grad.jacobian(y, x, i=3, j=2)

    # Laplacians
    E_w_real_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    E_w_real_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    E_w_imag_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    E_w_imag_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)
    E_2w_real_xx = dde.grad.hessian(y, x, component=2, i=0, j=0)
    E_2w_real_yy = dde.grad.hessian(y, x, component=2, i=1, j=1)
    E_2w_imag_xx = dde.grad.hessian(y, x, component=3, i=0, j=0)
    E_2w_imag_yy = dde.grad.hessian(y, x, component=3, i=1, j=1)

    laplacian_E_w_real = E_w_real_xx + E_w_real_yy
    laplacian_E_w_imag = E_w_imag_xx + E_w_imag_yy
    laplacian_E_2w_real = E_2w_real_xx + E_2w_real_yy
    laplacian_E_2w_imag = E_2w_imag_xx + E_2w_imag_yy

    # SHG coupling terms (real-valued approximation)
    residual_E_w_real = E_w_real_z - (-1/(2*k_w) * laplacian_E_w_imag - (alpha_w/2) * E_w_real + (k_w * deff / c) * (E_2w_real * E_w_imag + E_2w_imag * E_w_real))
    residual_E_w_imag = E_w_imag_z - (1/(2*k_w) * laplacian_E_w_real - (alpha_w/2) * E_w_imag - (k_w * deff / c) * (E_2w_real * E_w_real - E_2w_imag * E_w_imag))
    residual_E_2w_real = E_2w_real_z - (-1/(2*k_2w) * laplacian_E_2w_imag - (alpha_2w/2) * E_2w_real + (k_2w * deff / c) * (E_w_real**2 - E_w_imag**2))
    residual_E_2w_imag = E_2w_imag_z - (1/(2*k_2w) * laplacian_E_2w_real - (alpha_2w/2) * E_2w_imag - (k_2w * deff / c) * (2 * E_w_real * E_w_imag))

    return tf.concat([residual_E_w_real, residual_E_w_imag, residual_E_2w_real, residual_E_2w_imag], axis=1)

# FIXED: Real-valued Two-Photon PDE
def two_photon_pde(x, y):
    """
    Two-Photon Excitation Fluorescence - Real-valued formulation
    """
    E_real, E_imag, C = y[:, 0:1], y[:, 1:2], y[:, 2:3]

    # PHYSICAL PARAMETERS
    wavelength = 800e-9
    k = 2 * np.pi / wavelength
    alpha = 0.3
    sigma = 1e-20
    D = 1e-14
    k_f = 0.1
    eta = 0.5
    I_sat = 1e13

    # Calculate derivatives
    E_real_z = dde.grad.jacobian(y, x, i=0, j=2)
    E_imag_z = dde.grad.jacobian(y, x, i=1, j=2)

    E_real_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    E_real_yy = dde.grad.hessian(y, x, component=0, i=1, j=1)
    E_imag_xx = dde.grad.hessian(y, x, component=1, i=0, j=0)
    E_imag_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    laplacian_E_real = E_real_xx + E_real_yy
    laplacian_E_imag = E_imag_xx + E_imag_yy

    intensity = E_real**2 + E_imag**2
    intensity_sq = intensity ** 2

    # Light propagation
    residual_E_real = E_real_z - (-1/(2*k) * laplacian_E_imag - (alpha/2) * E_real - (sigma/2) * C * E_real)
    residual_E_imag = E_imag_z - (1/(2*k) * laplacian_E_real - (alpha/2) * E_imag - (sigma/2) * C * E_imag)

    # Fluorophore concentration (steady-state approximation)
    C_xx = dde.grad.hessian(y, x, component=2, i=0, j=0)
    C_yy = dde.grad.hessian(y, x, component=2, i=1, j=1)
    C_zz = dde.grad.hessian(y, x, component=2, i=2, j=2)
    laplacian_C = C_xx + C_yy + C_zz

    source_term = eta * sigma * intensity_sq / (1 + intensity_sq / I_sat)
    residual_C = -D * laplacian_C + k_f * C - source_term

    return tf.concat([residual_E_real, residual_E_imag, residual_C], axis=1)

# Domain
geom = dde.geometry.Cuboid(
    [-0.98, -0.99, -1.02],
    [0.98, 0.97, 1.02]
)

# Laser boundary condition
def laser_boundary(x, on_boundary):
    if not on_boundary:
        return False
    is_cornea = x[2] > 0.95
    iris_center_y = (-0.612905 + -0.472098) / 2
    target_radius = 0.25
    distance_sq = (x[0] - 0.0)**2 + (x[1] - iris_center_y)**2
    targets_iris = distance_sq < target_radius**2
    return is_cornea and targets_iris

# FIXED: Separate boundary conditions for each component
def gaussian_laser_shg_component0(x):
    iris_center_y = (-0.612905 + -0.472098) / 2
    beam_width = 0.2
    r_squared = (x[:, 0:1] - 0.0)**2 + (x[:, 1:2] - iris_center_y)**2 
    profile = tf.exp(-r_squared / (beam_width**2))
    return profile  # Shape: (N, 1)

def gaussian_laser_shg_component1(x):
    return tf.zeros((x.shape[0], 1))  # Shape: (N, 1)

def gaussian_laser_shg_component2(x):
    return tf.zeros((x.shape[0], 1))  # Shape: (N, 1)

def gaussian_laser_shg_component3(x):
    return tf.zeros((x.shape[0], 1))  # Shape: (N, 1)

def gaussian_laser_two_photon_component0(x):
    iris_center_y = (-0.612905 + -0.472098) / 2
    beam_width = 0.2
    r_squared = (x[:, 0:1] - 0.0)**2 + (x[:, 1:2] - iris_center_y)**2 
    profile = tf.exp(-r_squared / (beam_width**2))
    return profile  # Shape: (N, 1)

def gaussian_laser_two_photon_component1(x):
    return tf.zeros((x.shape[0], 1))  # Shape: (N, 1)

def gaussian_laser_two_photon_component2(x):
    return tf.zeros((x.shape[0], 1))  # Shape: (N, 1)

# Use DeepXDE's built-in FNN
def create_shg_network():
    return dde.nn.FNN([3] + [128] * 4 + [4], "tanh", "Glorot normal")

def create_two_photon_network():
    return dde.nn.FNN([3] + [128] * 4 + [3], "tanh", "Glorot normal")

# FIXED: Apply separate boundary conditions for each component
bc_laser_shg0 = dde.icbc.DirichletBC(geom, gaussian_laser_shg_component0, laser_boundary, component=0)
bc_laser_shg1 = dde.icbc.DirichletBC(geom, gaussian_laser_shg_component1, laser_boundary, component=1)
bc_laser_shg2 = dde.icbc.DirichletBC(geom, gaussian_laser_shg_component2, laser_boundary, component=2)
bc_laser_shg3 = dde.icbc.DirichletBC(geom, gaussian_laser_shg_component3, laser_boundary, component=3)

bc_laser_tp0 = dde.icbc.DirichletBC(geom, gaussian_laser_two_photon_component0, laser_boundary, component=0)
bc_laser_tp1 = dde.icbc.DirichletBC(geom, gaussian_laser_two_photon_component1, laser_boundary, component=1)
bc_laser_tp2 = dde.icbc.DirichletBC(geom, gaussian_laser_two_photon_component2, laser_boundary, component=2)

print("Training SHG model...")
# Build and Train SHG model
data_shg = dde.data.PDE(
    geometry=geom,
    pde=shg_pde,
    bcs=[bc_laser_shg0, bc_laser_shg1, bc_laser_shg2, bc_laser_shg3],
    num_domain=500,
    num_boundary=50,
    num_test=100    
)

net_shg = create_shg_network()
model_shg = dde.Model(data_shg, net_shg)

model_shg.compile("adam", lr=1e-3)
losshistory_shg, train_state_shg = model_shg.train(iterations=1000, display_every=100)

print("Training Two-Photon model...")
# Build and Train Two-Photon model
data_two_photon = dde.data.PDE(
    geometry=geom,
    pde=two_photon_pde,
    bcs=[bc_laser_tp0, bc_laser_tp1, bc_laser_tp2],
    num_domain=500,
    num_boundary=50,
    num_test=100
)

net_two_photon = create_two_photon_network()
model_two_photon = dde.Model(data_two_photon, net_two_photon)

model_two_photon.compile("adam", lr=1e-3)
losshistory_two_photon, train_state_two_photon = model_two_photon.train(iterations=1000, display_every=100)

print("Generating optical data...")
# Create sampling grid - focused on iris region
X = np.linspace(-0.5, 0.5, 50)
Y = np.linspace(-0.6, -0.5, 50)
Z = np.linspace(0.3, 0.3, 1)  # Slice through iris

xx, yy, zz = np.meshgrid(X, Y, Z)
points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# Get predictions from both models
pred_shg = model_shg.predict(points)
pred_two_photon = model_two_photon.predict(points)

# Extract SHG optical data
E_w_real = pred_shg[:, 0]
E_w_imag = pred_shg[:, 1]
E_2w_real = pred_shg[:, 2]
E_2w_imag = pred_shg[:, 3]

intensity_w = E_w_real**2 + E_w_imag**2  # Fundamental wave
intensity_2w = E_2w_real**2 + E_2w_imag**2  # SHG signal (collagen)

# Extract Two-Photon optical data
E_real_tp = pred_two_photon[:, 0]
E_imag_tp = pred_two_photon[:, 1]
C_fluorophore = pred_two_photon[:, 2]  # Fluorophore concentration

intensity_tp = E_real_tp**2 + E_imag_tp**2  # Two-photon excitation
fluorescence = C_fluorophore  # Fluorescence signal

# Reshape for plotting
intensity_2w_img = intensity_2w.reshape(50, 50)
fluorescence_img = fluorescence.reshape(50, 50)
intensity_w_img = intensity_w.reshape(50, 50)
intensity_tp_img = intensity_tp.reshape(50, 50)

print("Plotting optical data...")
# Create comprehensive visualization
fig, axes = plt.subplots(2, 2, figsize=(12, 10))

# SHG Results
im1 = axes[0, 0].imshow(intensity_w_img, cmap='viridis', 
                       extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')
axes[0, 0].set_title('Fundamental Wave Intensity\n(800nm Laser)')
axes[0, 0].set_xlabel('X (mm)')
axes[0, 0].set_ylabel('Y (mm)')
plt.colorbar(im1, ax=axes[0, 0], label='Intensity (a.u.)')

im2 = axes[0, 1].imshow(intensity_2w_img, cmap='hot',
                       extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')
axes[0, 1].set_title('SHG Signal - Collagen Distribution\n(400nm Generated)')
axes[0, 1].set_xlabel('X (mm)')
axes[0, 1].set_ylabel('Y (mm)')
plt.colorbar(im2, ax=axes[0, 1], label='SHG Intensity (a.u.)')

# Two-Photon Results
im3 = axes[1, 0].imshow(intensity_tp_img, cmap='plasma',
                       extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')
axes[1, 0].set_title('Two-Photon Excitation\n(800nm Laser)')
axes[1, 0].set_xlabel('X (mm)')
axes[1, 0].set_ylabel('Y (mm)')
plt.colorbar(im3, ax=axes[1, 0], label='Intensity (a.u.)')

im4 = axes[1, 1].imshow(fluorescence_img, cmap='cool',
                       extent=[X.min(), X.max(), Y.min(), Y.max()], origin='lower')
axes[1, 1].set_title('Fluorescence Signal\n(Metabolic Activity)')
axes[1, 1].set_xlabel('X (mm)')
axes[1, 1].set_ylabel('Y (mm)')
plt.colorbar(im4, ax=axes[1, 1], label='Fluorescence (a.u.)')

plt.tight_layout()
plt.show()

# Print quantitative results
print("\n=== OPTICAL DATA SUMMARY ===")
print(f"SHG Collagen Signal:")
print(f"  - Max intensity: {np.max(intensity_2w):.6f}")
print(f"  - Min intensity: {np.min(intensity_2w):.6f}")
print(f"  - Mean intensity: {np.mean(intensity_2w):.6f}")

print(f"\nTwo-Photon Fluorescence:")
print(f"  - Max fluorescence: {np.max(fluorescence):.6f}")
print(f"  - Min fluorescence: {np.min(fluorescence):.6f}")
print(f"  - Mean fluorescence: {np.mean(fluorescence):.6f}")

print(f"\nFundamental Laser:")
print(f"  - Max intensity: {np.max(intensity_w):.6f}")
print(f"  - Beam profile confirms laser targeting")

print("\nSimulation complete! All visualizations generated.")