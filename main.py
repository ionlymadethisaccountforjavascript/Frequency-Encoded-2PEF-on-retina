#IMSKBIIDI
import deepxde as dde
import torch
import trimesh
import numpy as np

import matplotlib.pyplot as plt
from scipy.spatial import cKDTree #prob not needed


import os
import tensorflow as tf


scene = trimesh.load('human_eye.glb')  
iris_mesh = scene.geometry['Eye_Iris_0']
bigbig_mesh = scene.to_geometry() #probs not needed

vertices = bigbig_mesh.vertices     #probably not needed
faces = bigbig_mesh.faces            #probs not needed
vertices = vertices.astype(np.float32)  # Convert to float32
faces = faces.astype(np.int32)          # Convert to int32



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

# figure made just to check, will do it thru more detail later on

# Figure
# Create figure
fig = plt.figure(figsize=(18, 6))

# -----------------------------
# Plot 1: Collagen Density
ax1 = fig.add_subplot(131, projection='3d')
for name, mesh in scene.geometry.items():
    verts = mesh.vertices
    prop = properties[name]
    ax1.scatter(verts[:,0], verts[:,1], verts[:,2],
                c=np.full(len(verts), prop['collagen_density']),
                cmap='viridis', s=2, alpha=0.7,
                vmin=0, vmax=1)  # <- scaled colormap
ax1.set_title('Collagen Density')
ax1.set_xlabel('X'); ax1.set_ylabel('Y'); ax1.set_zlabel('Z')
mappable1 = plt.cm.ScalarMappable(cmap='viridis')
mappable1.set_array([0,1])
fig.colorbar(mappable1, ax=ax1, shrink=0.5, label='Collagen Density')

# -----------------------------
# Plot 2: Melanin Content
ax2 = fig.add_subplot(132, projection='3d')
for name, mesh in scene.geometry.items():
    verts = mesh.vertices
    prop = properties[name]
    ax2.scatter(verts[:,0], verts[:,1], verts[:,2],
                c=np.full(len(verts), prop['melanin_content']),
                cmap='plasma', s=2, alpha=0.7,
                vmin=0, vmax=1)  # <- scaled colormap
ax2.set_title('Melanin Content')
ax2.set_xlabel('X'); ax2.set_ylabel('Y'); ax2.set_zlabel('Z')
mappable2 = plt.cm.ScalarMappable(cmap='plasma')
mappable2.set_array([0,1])
fig.colorbar(mappable2, ax=ax2, shrink=0.5, label='Melanin Content')

# -----------------------------
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



#defining pdes
def kerr_effect(x,y):
    '''kerr effect: 

    '''
    #physical quantities defined
    wavelength = 800e-9
    n2 = 2.4e-20
    A_eff = 1e-12
    alpha = 0.3#decide later
    gamma = (2*np.pi * n2)/(wavelength * A_eff)
    k = 2 * np.pi / wavelength
    E_real = y[:,0:1]
    E_imag = y[:,1:2]

    #defining first derivatives
    #propagation term(partial E/partial z)

    E_real_z = dde.grad.jacobian(y, x, i=0, j=2)
    E_imag_z = dde.grad.jacobian(y, x, i=1, j=2)

    #defining second derivatives
    E_real_xx = dde.grad.hessian(y, x, component=0, i=0, j=0)
    E_real_yy = dde.grad.hessian(y, x, component=0, i=0, j=1)
    E_imag_xx = dde.grad.hessian(y, x, component=1, i=1, j=0)
    E_imag_yy = dde.grad.hessian(y, x, component=1, i=1, j=1)

    #defining laplacian
    laplacian_E_real = E_real_xx + E_real_yy
    laplacian_E_imag = E_imag_xx + E_imag_yy

    #intensity:
    intensity = E_real**2 + E_imag**2

    kerr_nonlinear_term_real = -gamma * intensity * E_imag
    kerr_nonlinear_term_imag = gamma * intensity * E_real

    #linear absorption term
    absorption_real = (-alpha/2)*E_real
    absorption_imag = (-alpha/2)*E_imag

    #diffraction terms
    diffraction_real = - (1/(2*k)) * laplacian_E_imag
    diffraction_imag = (1/(2*k)) * laplacian_E_real

    #residual calculation
    residual_real = E_real_z - (diffraction_real + kerr_nonlinear_term_real + absorption_real)
    residual_imag = E_imag_z - (diffraction_imag + kerr_nonlinear_term_imag + absorption_imag)

    return [residual_real, residual_imag]

# second harmonic generation
def shg_pde(x, y):
    """
    Second Harmonic Generation (SHG) - Coupled Wave Equations.
    x: coordinates [x, y, z]
    y: network output [E_w_real, E_w_imag, E_2w_real, E_2w_imag]
    """
    # Split the output into fundamental and second harmonic fields
    E_w_real, E_w_imag, E_2w_real, E_2w_imag = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    E_w = tf.complex(E_w_real, E_w_imag)
    E_2w = tf.complex(E_2w_real, E_2w_imag)

    # PHYSICAL PARAMETERS (You can adjust these)
    wavelength = 800e-9
    k_w = 2 * np.pi / wavelength
    k_2w = 2 * np.pi / (wavelength / 2)
    deff = 1e-12
    alpha_w = 0.1
    alpha_2w = 0.1
    Delta_k = 2 * k_w - k_2w
    c = 3e8

    # Calculate derivatives for the fundamental wave (ω)
    E_w_z = dde.grad.jacobian(y, x, i=0, j=2) + 1j * dde.grad.jacobian(y, x, i=1, j=2)
    E_w_xx = dde.grad.hessian(y, x, i=0, j=0) + 1j * dde.grad.hessian(y, x, i=1, j=0)
    E_w_yy = dde.grad.hessian(y, x, i=0, j=1) + 1j * dde.grad.hessian(y, x, i=1, j=1)
    laplacian_w = E_w_xx + E_w_yy

    # Calculate derivatives for the second harmonic wave (2ω)
    E_2w_z = dde.grad.jacobian(y, x, i=2, j=2) + 1j * dde.grad.jacobian(y, x, i=3, j=2)
    E_2w_xx = dde.grad.hessian(y, x, i=2, j=0) + 1j * dde.grad.hessian(y, x, i=3, j=0)
    E_2w_yy = dde.grad.hessian(y, x, i=2, j=1) + 1j * dde.grad.hessian(y, x, i=3, j=1)
    laplacian_2w = E_2w_xx + E_2w_yy

    # PDE for the Fundamental Wave (ω)
    RHS_w = (1j/(2*k_w)) * laplacian_w - (alpha_w/2) * E_w + 1j * (k_w * deff / c) * E_2w * tf.math.conj(E_w) * tf.exp(-1j * Delta_k * x[:, 2:3])
    residual_w = E_w_z - RHS_w

    # PDE for the Second Harmonic Wave (2ω)
    RHS_2w = (1j/(2*k_2w)) * laplacian_2w - (alpha_2w/2) * E_2w + 1j * (k_2w * deff / c) * E_w * E_w * tf.exp(1j * Delta_k * x[:, 2:3])
    residual_2w = E_2w_z - RHS_2w

    # Return the residuals for both real and imaginary parts
    return tf.concat([tf.math.real(residual_w), tf.math.imag(residual_w), tf.math.real(residual_2w), tf.math.imag(residual_2w)], axis=1)

def two_pef_pde(x, y):

    E_real, E_imag, C = y[:, 0:1], y[:, 1:2], y[:, 2:3]
    E = tf.complex(E_real, E_imag)

    # PHYSICAL PARAMETERS
    wavelength = 800e-9
    k = 2 * np.pi / wavelength
    alpha = 0.3
    sigma = 1e-20
    D = 1e-14
    k_f = 0.1
    eta = 0.5
    I_sat = 1e13

    # light propagation pde (i defined earlier too wth)
    E_z = dde.grad.jacobian(y, x, i=0, j=2) + 1j * dde.grad.jacobian(y, x, i=1, j=2)
    E_xx = dde.grad.hessian(y, x, i=0, j=0) + 1j * dde.grad.hessian(y, x, i=1, j=0)
    E_yy = dde.grad.hessian(y, x, i=0, j=1) + 1j * dde.grad.hessian(y, x, i=1, j=1)
    laplacian_E = E_xx + E_yy

    intensity = tf.math.real(E * tf.math.conj(E))
    intensity_sq = intensity ** 2

    RHS_E = (1j/(2*k)) * laplacian_E - ( (alpha/2) + (sigma/2) * C ) * E
    residual_E = E_z - RHS_E

    # fluorophore concentration pde
    C_t = dde.grad.jacobian(y, x, i=2, j=3)
    C_xx = dde.grad.hessian(y, x, i=2, j=0)
    C_yy = dde.grad.hessian(y, x, i=2, j=1)
    laplacian_C = C_xx + C_yy

    source_term = eta * sigma * intensity_sq / (1 + intensity_sq / I_sat)
    RHS_C = D * laplacian_C - k_f * C + source_term
    residual_C = C_t - RHS_C

    return tf.concat([tf.math.real(residual_E), tf.math.imag(residual_E), residual_C], axis=1)

#doomainz
geom = dde.geometry.Cuboid(
    [-0.54, -0.62, -0.54],   # Tight around iris
    [ 0.54, -0.47,  0.54]    # Your exact iris bounds
)

# FIXED laser boundary
def laser_boundary(x, on_boundary):
    if not on_boundary:
        return False
    # Laser hits FRONT of iris (not cornea)
    is_front = x[2] > 0.3  # Front surface of iris
    # Target center region
    iris_center_y = (-0.612905 + -0.472098) / 2
    target_radius = 0.2
    # FIXED: Use squared distance correctly
    distance_sq = (x[0] - 0.0)**2 + (x[1] - iris_center_y)**2
    targets_center = distance_sq < target_radius**2
    return is_front and targets_center

# FIXED Gaussian laser
def gaussian_laser(x):
    iris_center_y = (-0.612905 + -0.472098) / 2
    beam_width = 0.15
    # FIXED: Correct squared distance calculation
    r_squared = (x[:, 0:1] - 0.0)**2 + (x[:, 1:2] - iris_center_y)**2
    profile = tf.exp(-r_squared / (beam_width**2))
    
    return tf.concat([profile, tf.zeros_like(profile), 
                     tf.zeros_like(profile), tf.zeros_like(profile)], axis=1)

# Apply boundary condition
bc_laser = dde.icbc.DirichletBC(geom, gaussian_laser, laser_boundary)

def create_shg_layers():
    input_dim = 3  
    output_dim = 4  
    
    network = dde.nn.FNN(
        [input_dim] + [256] * 8 + [output_dim],
        "tanh",
        "Glorot normal"
    )
    return network

def create_twophoton_layers():
    input_dim = 4  
    output_dim = 3  
    
    network = dde.nn.FNN(
        [input_dim] + [192] * 8 + [output_dim],
        "tanh", 
        "He normal"
    )
    return network

def create_residual_shg():
    input_dim = 3
    output_dim = 4
    
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    x = tf.keras.layers.Dense(256, activation="swish")(inputs)
    for _ in range(6):
        residual = x
        x = tf.keras.layers.Dense(256, activation="swish")(x)
        x = tf.keras.layers.Dense(256, activation="swish")(x)
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.LayerNormalization()(x)
    
    outputs = tf.keras.layers.Dense(output_dim)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

def create_residual_twophoton():
    input_dim = 4
    output_dim = 3
    
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    x = tf.keras.layers.Dense(192, activation="swish")(inputs)
    for _ in range(5):
        residual = x
        x = tf.keras.layers.Dense(192, activation="swish")(x)
        x = tf.keras.layers.Dense(192, activation="swish")(x)
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.LayerNormalization()(x)
    
    outputs = tf.keras.layers.Dense(output_dim)(x)
    return tf.keras.Model(inputs=inputs, outputs=outputs)

   

# -----------------------
# STEP 6: Build and Train model
# -----------------------
data = dde.data.PDE(
    geom=geom,
    pde=shg_pde,
    bcs=[bc_laser],
    num_domain=20000,
    num_boundary=2000,
    num_test=2000
    )

net = create_residual_shg()
model = dde.Model(data, net)
model.compile("adam", lr=1e-4)
losshistory, train_state = model.train(epochs=10000)

# Optional: LBFGS for extra accuracy after Adam
model.compile("L-BFGS")
model.train()

X = np.linspace(-0.533915, 0.533915, 100)
Y = np.linspace(-0.612905, -0.472098, 100)
Z = np.linspace(0.4, 0.4, 1)  # thin slice at ~front iris

xx, yy, zz = np.meshgrid(X, Y, Z)
points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

# Predict SHG field
pred = model.predict(points)
E_real = pred[:, 0]
E_imag = pred[:, 1]
intensity = E_real**2 + E_imag**2

# Plot 2D SHG intensity map (X vs Y at Z ≈ 0.4)
plt.figure(figsize=(8, 6))
plt.imshow(intensity.reshape(100, 100),
           cmap='hot',
           extent=[X.min(), X.max(), Y.min(), Y.max()],
           origin='lower')  # flip Y-axis to match coord orientation

plt.title("SHG Intensity (Slice at Z ≈ 0.4)")
plt.xlabel("X")
plt.ylabel("Y")
plt.colorbar(label="Intensity")
plt.tight_layout()
plt.show()

    
