
#IMSKBIIDI
import deepxde
import torch
import trimesh
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree #prob not needed


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

