import deepxde as dde
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
import os
import tensorflow as tf

# Set DeepXDE configuration
dde.config.set_default_float("float64")
dde.config.set_default_backend("tensorflow")

# Load and inspect the eye mesh
current_dir = os.path.dirname(os.path.abspath(__file__))
glb_file = os.path.join(current_dir, "human_eye.glb")

try:
    scene = trimesh.load(glb_file)
    print(f"Scene loaded successfully: {len(scene.geometry)} meshes")
    
    if isinstance(scene, trimesh.Scene):
        for name, mesh in scene.geometry.items():
            print(f"{name}: {len(mesh.vertices)} vertices, {len(mesh.faces)} faces")
    else:
        print(f"Single mesh: {len(scene.vertices)} vertices, {len(scene.faces)} faces")
        
except Exception as e:
    print(f"Error loading mesh: {e}")
    # Create a simple fallback geometry for testing
    print("Using fallback geometry...")
    scene = None

# Define material properties
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

# Visualization (optional)
def plot_eye_properties(scene):
    if scene is None:
        return
        
    fig = plt.figure(figsize=(18, 6))
    
    # Collagen Density
    ax1 = fig.add_subplot(131, projection='3d')
    for name, mesh in scene.geometry.items():
        verts = mesh.vertices
        prop = properties.get(name, {})
        ax1.scatter(verts[:,0], verts[:,1], verts[:,2],
                    c=np.full(len(verts), prop.get('collagen_density', 0.5)),
                    cmap='viridis', s=2, alpha=0.7, vmin=0, vmax=1)
    ax1.set_title('Collagen Density')
    
    # Melanin Content  
    ax2 = fig.add_subplot(132, projection='3d')
    for name, mesh in scene.geometry.items():
        verts = mesh.vertices
        prop = properties.get(name, {})
        ax2.scatter(verts[:,0], verts[:,1], verts[:,2],
                    c=np.full(len(verts), prop.get('melanin_content', 0.5)),
                    cmap='plasma', s=2, alpha=0.7, vmin=0, vmax=1)
    ax2.set_title('Melanin Content')
    
    # Iris Region
    ax3 = fig.add_subplot(133, projection='3d')
    colors = {True: 'red', False: 'blue'}
    for name, mesh in scene.geometry.items():
        verts = mesh.vertices
        prop = properties.get(name, {})
        ax3.scatter(verts[:,0], verts[:,1], verts[:,2],
                    color=colors.get(prop.get('is_iris', False), 'gray'), 
                    s=2, alpha=0.7, label=name)
    ax3.set_title('Iris Region')
    ax3.legend()
    
    plt.tight_layout()
    plt.show()

# Uncomment to plot
# plot_eye_properties(scene)

# Define PDEs
def shg_pde(x, y):
    """
    Second Harmonic Generation (SHG) - Coupled Wave Equations.
    x: coordinates [x, y, z]
    y: network output [E_w_real, E_w_imag, E_2w_real, E_2w_imag]
    """
    # Split the output into fundamental and second harmonic fields
    E_w_real, E_w_imag, E_2w_real, E_2w_imag = y[:, 0:1], y[:, 1:2], y[:, 2:3], y[:, 3:4]
    
    # PHYSICAL PARAMETERS
    wavelength = 800e-9
    k_w = 2 * np.pi / wavelength
    k_2w = 2 * np.pi / (wavelength / 2)
    deff = 1e-12
    alpha_w = 0.1
    alpha_2w = 0.1
    Delta_k = 2 * k_w - k_2w
    c = 3e8

    # Calculate derivatives for the fundamental wave (ω)
    E_w_z_real = dde.grad.jacobian(y, x, i=0, j=2)
    E_w_z_imag = dde.grad.jacobian(y, x, i=1, j=2)
    
    E_w_xx_real = dde.grad.hessian(y, x, component=0, i=0, j=0)
    E_w_xx_imag = dde.grad.hessian(y, x, component=1, i=1, j=0)
    E_w_yy_real = dde.grad.hessian(y, x, component=0, i=0, j=1) 
    E_w_yy_imag = dde.grad.hessian(y, x, component=1, i=1, j=1)
    
    laplacian_w_real = E_w_xx_real + E_w_yy_real
    laplacian_w_imag = E_w_xx_imag + E_w_yy_imag

    # Calculate derivatives for the second harmonic wave (2ω)
    E_2w_z_real = dde.grad.jacobian(y, x, i=2, j=2)
    E_2w_z_imag = dde.grad.jacobian(y, x, i=3, j=2)
    
    E_2w_xx_real = dde.grad.hessian(y, x, component=2, i=2, j=0)
    E_2w_xx_imag = dde.grad.hessian(y, x, component=3, i=3, j=0)
    E_2w_yy_real = dde.grad.hessian(y, x, component=2, i=2, j=1)
    E_2w_yy_imag = dde.grad.hessian(y, x, component=3, i=3, j=1)
    
    laplacian_2w_real = E_2w_xx_real + E_2w_yy_real
    laplacian_2w_imag = E_2w_xx_imag + E_2w_yy_imag

    # Nonlinear terms
    intensity_w = E_w_real**2 + E_w_imag**2
    nonlinear_real_w = (k_w * deff / c) * (E_2w_real * E_w_real + E_2w_imag * E_w_imag) * tf.cos(Delta_k * x[:, 2:3]) + (k_w * deff / c) * (E_2w_imag * E_w_real - E_2w_real * E_w_imag) * tf.sin(Delta_k * x[:, 2:3])
    nonlinear_imag_w = (k_w * deff / c) * (E_2w_imag * E_w_real - E_2w_real * E_w_imag) * tf.cos(Delta_k * x[:, 2:3]) - (k_w * deff / c) * (E_2w_real * E_w_real + E_2w_imag * E_w_imag) * tf.sin(Delta_k * x[:, 2:3])

    # PDE for the Fundamental Wave (ω)
    diffraction_real_w = - (1/(2*k_w)) * laplacian_w_imag
    diffraction_imag_w = (1/(2*k_w)) * laplacian_w_real
    absorption_real_w = (-alpha_w/2) * E_w_real
    absorption_imag_w = (-alpha_w/2) * E_w_imag
    
    residual_w_real = E_w_z_real - (diffraction_real_w + nonlinear_real_w + absorption_real_w)
    residual_w_imag = E_w_z_imag - (diffraction_imag_w + nonlinear_imag_w + absorption_imag_w)

    # PDE for the Second Harmonic Wave (2ω)  
    nonlinear_real_2w = (k_2w * deff / c) * (E_w_real**2 - E_w_imag**2) * tf.cos(Delta_k * x[:, 2:3]) - (k_2w * deff / c) * (2 * E_w_real * E_w_imag) * tf.sin(Delta_k * x[:, 2:3])
    nonlinear_imag_2w = (k_2w * deff / c) * (2 * E_w_real * E_w_imag) * tf.cos(Delta_k * x[:, 2:3]) + (k_2w * deff / c) * (E_w_real**2 - E_w_imag**2) * tf.sin(Delta_k * x[:, 2:3])

    diffraction_real_2w = - (1/(2*k_2w)) * laplacian_2w_imag
    diffraction_imag_2w = (1/(2*k_2w)) * laplacian_2w_real
    absorption_real_2w = (-alpha_2w/2) * E_2w_real
    absorption_imag_2w = (-alpha_2w/2) * E_2w_imag
    
    residual_2w_real = E_2w_z_real - (diffraction_real_2w + nonlinear_real_2w + absorption_real_2w)
    residual_2w_imag = E_2w_z_imag - (diffraction_imag_2w + nonlinear_imag_2w + absorption_imag_2w)

    return tf.concat([residual_w_real, residual_w_imag, residual_2w_real, residual_2w_imag], axis=1)

# Define geometry (adjusted to be more reasonable)
geom = dde.geometry.Cuboid(
    [-0.5, -0.6, -0.5],   # min coordinates
    [ 0.5, -0.5,  0.5]    # max coordinates
)

# Laser boundary condition
def laser_boundary(x, on_boundary):
    if not on_boundary:
        return False
    # Laser hits front surface
    is_front = np.isclose(x[2], 0.5, atol=0.1)  # Front surface
    # Target center region
    iris_center_y = -0.55
    target_radius = 0.2
    distance_sq = (x[0] - 0.0)**2 + (x[1] - iris_center_y)**2
    targets_center = distance_sq < target_radius**2
    return is_front and targets_center

# Gaussian laser profile
def gaussian_laser(x):
    iris_center_y = -0.55
    beam_width = 0.15
    r_squared = (x[:, 0:1] - 0.0)**2 + (x[:, 1:2] - iris_center_y)**2
    profile = tf.exp(-r_squared / (beam_width**2))
    
    return tf.concat([profile, tf.zeros_like(profile), 
                     tf.zeros_like(profile), tf.zeros_like(profile)], axis=1)

# Create residual network for SHG
def create_residual_shg():
    input_dim = 3
    output_dim = 4
    
    inputs = tf.keras.layers.Input(shape=(input_dim,))
    
    x = tf.keras.layers.Dense(128, activation="swish")(inputs)
    for _ in range(4):  # Reduced for stability
        residual = x
        x = tf.keras.layers.Dense(128, activation="swish")(x)
        x = tf.keras.layers.Dense(128, activation="swish")(x)
        x = tf.keras.layers.Add()([x, residual])
        x = tf.keras.layers.LayerNormalization()(x)
    
    outputs = tf.keras.layers.Dense(output_dim)(x)
    
    return tf.keras.Model(inputs=inputs, outputs=outputs)

# Create model directory
model_dir = "saved_models"
os.makedirs(model_dir, exist_ok=True)

# Build and train the model
def train_model():
    print("Setting up PDE and boundary conditions...")
    
    bc_laser = dde.icbc.DirichletBC(geom, gaussian_laser, laser_boundary)
    
    data = dde.data.PDE(
        geom=geom,
        pde=shg_pde,
        bcs=[bc_laser],
        num_domain=1000,  # Reduced for faster training
        num_boundary=100,
        num_test=200
    )

    print("Creating neural network...")
    net = create_residual_shg()
    model = dde.Model(data, net)
    
    # Model checkpoint callback
    checkpoint_path = os.path.join(model_dir, "shg_model")
    checkpointer = dde.callbacks.ModelCheckpoint(
        checkpoint_path, 
        verbose=1, 
        save_better_only=True, 
        period=500
    )
    
    print("Compiling model...")
    model.compile("adam", lr=1e-4)
    
    print("Starting training...")
    losshistory, train_state = model.train(
        epochs=2000,  # Reduced for testing
        callbacks=[checkpointer],
        display_every=100
    )
    
    # Save final model
    model.save(checkpoint_path + "-final")
    
    print("Training completed!")
    return model, losshistory, train_state

# Prediction function
def predict_and_plot(model_path=None):
    if model_path is None:
        model_path = os.path.join(model_dir, "shg_model-final")
    
    print(f"Loading model from {model_path}...")
    
    # Recreate the network and model structure
    net = create_residual_shg()
    bc_laser = dde.icbc.DirichletBC(geom, gaussian_laser, laser_boundary)
    
    data = dde.data.PDE(
        geom=geom,
        pde=shg_pde, 
        bcs=[bc_laser],
        num_domain=10,
        num_boundary=10,
        num_test=10
    )
    
    model = dde.Model(data, net)
    model.compile("adam", lr=1e-4)
    
    # Load the trained weights
    try:
        model.restore(model_path)
        print("Model restored successfully!")
    except Exception as e:
        print(f"Error restoring model: {e}")
        print("You may need to train the model first.")
        return None

    # Create prediction grid
    print("Generating prediction grid...")
    X = np.linspace(-0.5, 0.5, 50)
    Y = np.linspace(-0.6, -0.5, 50) 
    Z = np.array([0.4])  # Fixed Z slice
    
    xx, yy, zz = np.meshgrid(X, Y, Z)
    points = np.vstack([xx.ravel(), yy.ravel(), zz.ravel()]).T

    print("Making predictions...")
    pred = model.predict(points)
    E_real = pred[:, 0]
    E_imag = pred[:, 1]
    intensity = E_real**2 + E_imag**2

    # Plot results
    print("Plotting results...")
    plt.figure(figsize=(10, 8))
    
    plt.subplot(2, 2, 1)
    plt.imshow(intensity.reshape(len(Y), len(X)),
               cmap='hot', extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', aspect='auto')
    plt.title("SHG Intensity (Z = 0.4)")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.colorbar(label="Intensity")
    
    plt.subplot(2, 2, 2)
    plt.imshow(E_real.reshape(len(Y), len(X)),
               cmap='RdBu_r', extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', aspect='auto')
    plt.title("Real Part of E Field")
    plt.colorbar()
    
    plt.subplot(2, 2, 3)
    plt.imshow(E_imag.reshape(len(Y), len(X)),
               cmap='RdBu_r', extent=[X.min(), X.max(), Y.min(), Y.max()],
               origin='lower', aspect='auto')
    plt.title("Imaginary Part of E Field")
    plt.colorbar()
    
    plt.subplot(2, 2, 4)
    # Show the laser input profile for reference
    test_points = np.array([[x, -0.55, 0.5] for x in X])
    laser_profile = gaussian_laser(test_points)[:, 0]
    plt.plot(X, laser_profile, 'g-', linewidth=2)
    plt.title("Laser Input Profile")
    plt.xlabel("X position")
    plt.ylabel("Intensity")
    
    plt.tight_layout()
    plt.show()
    
    return intensity, E_real, E_imag

# Main execution
if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='SHG Simulation for Eye Tissue')
    parser.add_argument('--mode', choices=['train', 'predict', 'both'], default='both',
                       help='Run training, prediction, or both')
    parser.add_argument('--model_path', type=str, default=None,
                       help='Path to saved model for prediction')
    
    args = parser.parse_args()
    
    if args.mode in ['train', 'both']:
        print("=== TRAINING MODE ===")
        trained_model, loss_history, train_state = train_model()
    
    if args.mode in ['predict', 'both']:
        print("\n=== PREDICTION MODE ===")
        results = predict_and_plot(args.model_path)
        
        if results is not None:
            intensity, E_real, E_imag = results
            print(f"Prediction completed! Intensity range: {intensity.min():.2e} to {intensity.max():.2e}")