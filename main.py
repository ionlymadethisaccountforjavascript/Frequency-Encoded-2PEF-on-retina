#IMSKBIIDI
import deepxde as dde
import torch
import trimesh
import numpy as np
import os
import tensorflow as tf

scene = trimesh.load('human_eye.glb')  
bigbig_mesh = scene.to_geometry()

vertices = bigbig_mesh.vertices  
faces = bigbig_mesh.faces       

print(f"Geometry data: {vertices.shape}, {faces.shape}")

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