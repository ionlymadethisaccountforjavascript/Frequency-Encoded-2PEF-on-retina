import trimesh
import numpy as np

scene = trimesh.load('human_eye.glb')  
bigbig_mesh = scene.to_geometry()

vertices = bigbig_mesh.vertices  
faces = bigbig_mesh.faces       

print(f"Geometry data: {vertices.shape}, {faces.shape}")
