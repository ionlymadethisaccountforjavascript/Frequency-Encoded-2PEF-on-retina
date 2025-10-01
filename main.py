import trimesh
import os

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