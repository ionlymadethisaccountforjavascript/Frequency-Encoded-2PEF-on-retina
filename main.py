
#IMSKBIIDI
import deepxde
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
