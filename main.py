#IMSKBIIDI
import deepxde as dde
import torch
import trimesh
import numpy as np
import os

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
