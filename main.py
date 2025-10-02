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

'''
print(f"Geometry data: {vertices.shape}, {faces.shape}")
print(f"X: {vertices[:,0].min():.3f} to {vertices[:,0].max():.3f}")
print(f"Y: {vertices[:,1].min():.3f} to {vertices[:,1].max():.3f}")  
print(f"Z: {vertices[:,2].min():.3f} to {vertices[:,2].max():.3f}")



tree = cKDTree(iris_mesh.vertices.astype(np.float32))
distances,_ = tree.query(vertices, k=1)
tolerance = 1e-2
properties['is_iris'] = distances<tolerance

properties['collagen_density'][properties['is_iris']] = 0.8
properties['melanin_content'][properties['is_iris']] = 0.7
properties['collagen_density'][~properties['is_iris']] = 0.2
properties['melanin_content'][~properties['is_iris']] = 0.3
'''
'''
for i,vertex in enumerate(vertices):
    is_iris = False
    for iris_vertex in iris_mesh.vertices.astype(np.float32):
        if np.allclose(vertex, iris_vertex, atol=1e-5):
            is_iris = True
            break
    
    if is_iris:
        properties['is_iris'][i] = True
        properties['collagen_density'][i] = 0.8
        properties['melanin_content'][i] = 0.7
    else:
        properties['is_iris'][i] = False
        properties['collagen_density'][i] = 0.2
        properties['melanin_content'][i] = 0.3
print("=== TESTING PROPERTY ASSIGNMENT ===")
'''
'''
# Test 1: Check counts
print(f"📊 Total vertices: {len  (vertices)}")
print(f"🎯 Iris vertices found: {np.sum(properties['is_iris'])}")
print(f"⚪ Eyeball vertices: {np.sum(~properties['is_iris'])}")

# Test 2: Check value ranges
print(f"📈 Collagen range: {properties['collagen_density'].min():.1f} to {properties['collagen_density'].max():.1f}")
print(f"🎨 Melanin range: {properties['melanin_content'].min():.1f} to {properties['melanin_content'].max():.1f}")

# Test 3: Verify iris has correct values
iris_mask = properties['is_iris']
print(f"🔍 Iris collagen average: {properties['collagen_density'][iris_mask].mean():.2f}")
print(f"🔍 Iris melanin average: {properties['melanin_content'][iris_mask].mean():.2f}")

# Test 4: Verify eyeball has correct values  
eyeball_mask = ~properties['is_iris']
print(f"🔍 Eyeball collagen average: {properties['collagen_density'][eyeball_mask].mean():.2f}")
print(f"🔍 Eyeball melanin average: {properties['melanin_content'][eyeball_mask].mean():.2f}")

# Test 5: Visual inspection
plt.figure(figsize=(15, 5))
all_vertices = vertices
# Plot 1: Iris detection
plt.subplot(1, 3, 1)
plt.scatter(all_vertices[:,0], all_vertices[:,1], 
           c=properties['is_iris'], cmap='coolwarm', alpha=0.7, s=10)
plt.title('Iris Detection\nRed = Iris, Blue = Eyeball')
plt.colorbar()

# Plot 2: Collagen distribution
plt.subplot(1, 3, 2)
plt.scatter(all_vertices[:,0], all_vertices[:,1], 
           c=properties['collagen_density'], cmap='viridis', alpha=0.7, s=10)
plt.title('Collagen Density')
plt.colorbar()

# Plot 3: Melanin distribution
plt.subplot(1, 3, 3)
plt.scatter(all_vertices[:,0], all_vertices[:,1], 
           c=properties['melanin_content'], cmap='hot', alpha=0.7, s=10)
plt.title('Melanin Content')
plt.colorbar()

plt.tight_layout()
plt.savefig('property_test_results.png', dpi=150)
plt.show()

print("✅ Saved visualization: 'property_test_results.png'")


'''

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