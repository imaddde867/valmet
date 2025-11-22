import numpy as np
from plyfile import PlyData
import viser
from PIL import Image
import torch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load PLY file
print("Loading PLY...")
plydata = PlyData.read("assets/pointclouds/pilot_plant_devices.ply")
vertex = plydata['vertex']

# Extract arrays
means_np = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T
scales_np = np.vstack([vertex['scale_0'], vertex['scale_1'], vertex['scale_2']]).T
quats_np = np.vstack([vertex['rot_0'], vertex['rot_1'], vertex['rot_2'], vertex['rot_3']]).T
opacities_np = np.array(vertex['opacity'])
sh_dc_np = np.vstack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']]).T

# Torch tensors
means = torch.from_numpy(means_np).float().to(device)
scales = torch.from_numpy(scales_np).float().to(device)
quats = torch.from_numpy(quats_np).float().to(device)
opacities = torch.from_numpy(opacities_np).float().to(device)
sh_dc = torch.from_numpy(sh_dc_np).float().to(device)

# Convert SH to RGB
SH_C0 = 0.28209479177387814
colors = torch.clamp(sh_dc * SH_C0 + 0.5, 0, 1)

print(f"Loaded {len(means)} Gaussians")

img_width = 800
img_height = 600

server = viser.ViserServer()


# ---------------------------------------------------------
# CPU-only fallback rasterizer (MAC-SAFE)
# ---------------------------------------------------------
def simple_cpu_render(means, colors, camera_position, W, H):
    """Very simple CPU renderer: nearest splat projection."""
    img = np.zeros((H, W, 3), dtype=np.float32)

    # Project points
    pts = means.cpu().numpy()
    cols = colors.cpu().numpy()

    # Basic perspective projection
    cam = np.array(camera_position)
    rel = pts - cam

    z = rel[:, 2] + 1e-5
    x_proj = (rel[:, 0] / z) * 400 + W / 2
    y_proj = (rel[:, 1] / z) * 400 + H / 2

    for i in range(len(pts)):
        x = int(x_proj[i])
        y = int(y_proj[i])
        if 0 <= x < W and 0 <= y < H:
            img[y, x] = cols[i]

    return (img * 255).astype(np.uint8)


def render_from_camera(camera_position):
    """Safely attempts gsplat, but falls back to CPU renderer."""
    try:
        import gsplat

        print("Attempting gsplat render (may fail on macOS)...")

        # Force CPU backend (still no rasterizer shipped)
        if hasattr(gsplat, "set_backend"):
            gsplat.set_backend("cpu")

        # This WILL fail on macOS — no rendering_cpu module in your build
        from gsplat.rendering_cpu import rasterization  # noqa

        raise RuntimeError("Your gsplat build does not include CPU rasterizer.")

    except Exception:
        print("Using simplified CPU fallback renderer instead.")
        img = simple_cpu_render(means, colors, camera_position, img_width, img_height)
        return img


# Camera = center + distance
center = means_np.mean(axis=0)
bbox_size = means_np.max(axis=0) - means_np.min(axis=0)
distance = np.linalg.norm(bbox_size) * 1.5

default_camera_pos = center + np.array([0, 0, distance])

print(f"\nScene center: {center}")
print(f"Camera distance: {distance}")
print(f"Rendering from position: {default_camera_pos}\n")

# Render
img = render_from_camera(default_camera_pos)

Image.fromarray(img).save("rendered_output.png")
print("Render saved to: rendered_output.png")

# Add point cloud to viser viewer
colors_np = (sh_dc_np * SH_C0 + 0.5).clip(0, 1)
colors_uint8 = (colors_np * 255).astype(np.uint8)

server.scene.add_point_cloud(
    "/gaussians",
    points=means_np,
    colors=colors_uint8,
    point_size=0.003
)

print("\nViewer running at http://localhost:8080")

import time
while True:
    time.sleep(1)
