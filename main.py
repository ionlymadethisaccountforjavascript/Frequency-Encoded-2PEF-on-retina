import trimesh
import os

def render_glb_with_trimesh():
    """Render GLB file using Trimesh"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    glb_file = os.path.join(current_dir, "default_eye_ball.glb")
    
    if not os.path.exists(glb_file):
        print(f"❌ Error: {glb_file} not found!")
        print("Please make sure 'default_eye_ball.glb' is in the same folder as this script")
        return False
    
    try:
        print("📥 Loading GLB file...")
        # Load the GLB file (may contain multiple meshes/scenes)
        scene = trimesh.load(glb_file)
        
        # Check if it's a scene or single mesh
        if isinstance(scene, trimesh.Scene):
            print(f"✅ Scene loaded: {len(scene.geometry)} meshes")
            # Show the entire scene
            scene.show()
        else:
            # It's a single mesh
            print(f"✅ Mesh loaded: {len(scene.vertices)} vertices, {len(scene.faces)} faces")
            scene.show()
        
        # Optional: Save screenshot
        try:
            screenshot_path = os.path.join(current_dir, "eye_glb_render.png")
            
            if isinstance(scene, trimesh.Scene):
                png_data = scene.save_image(resolution=[1920, 1080], visible=True)
            else:
                png_data = scene.scene().save_image(resolution=[1920, 1080], visible=True)
            
            with open(screenshot_path, 'wb') as f:
                f.write(png_data)
            
            print(f"📸 Screenshot saved: {screenshot_path}")
        except Exception as screenshot_error:
            print(f"⚠️  Could not save screenshot: {screenshot_error}")
        
        return True
        
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

def quick_glb_view():
    """Even simpler GLB viewer"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    glb_file = os.path.join(current_dir, "default_eye_ball.glb")
    
    if not os.path.exists(glb_file):
        print(f"❌ File not found: {glb_file}")
        return False
    
    try:
        # Simple one-liner to view GLB
        trimesh.load(glb_file).show()
        return True
    except Exception as e:
        print(f"❌ Failed: {e}")
        return False

if __name__ == "__main__":
    print("🎬 Trimesh GLB Renderer")
    print("=" * 40)
    print("Rendering: default_eye_ball.glb")
    
    success = render_glb_with_trimesh()
    
    if success:
        print("\n🎉 Render complete!")
    else:
        print("\n🔄 Trying simple viewer...")
        quick_glb_view()