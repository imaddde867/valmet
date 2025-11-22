import numpy as np
from plyfile import PlyData
import viser
from PIL import Image
import torch

from device_utils import pick_device


device = pick_device()
print(f"Using device: {device}")
if device.type == "mps":
    print("Metal / MPS acceleration enabled.")

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


def torch_projective_render(
    means_t: torch.Tensor,
    colors_t: torch.Tensor,
    camera_position,
    W: int,
    H: int,
    focal_length: float = 400.0,
) -> np.ndarray:
    """Render by projecting splats using torch on the active device."""
    cam = torch.tensor(camera_position, dtype=means_t.dtype, device=device)
    rel = means_t - cam
    z = rel[:, 2].clamp(min=1e-5)
    x_proj = (rel[:, 0] / z) * focal_length + W / 2
    y_proj = (rel[:, 1] / z) * focal_length + H / 2

    x_idx = x_proj.round().long()
    y_idx = y_proj.round().long()

    valid = (
        (x_idx >= 0)
        & (x_idx < W)
        & (y_idx >= 0)
        & (y_idx < H)
        & torch.isfinite(z)
    )

    if not torch.any(valid):
        return np.zeros((H, W, 3), dtype=np.uint8)

    x_idx = x_idx[valid]
    y_idx = y_idx[valid]
    cols = colors_t[valid]
    lin_idx = y_idx * W + x_idx

    img_flat = torch.zeros((H * W, 3), dtype=cols.dtype, device=device)
    expanded_idx = lin_idx.unsqueeze(-1).expand(-1, 3)
    img_flat.scatter_add_(0, expanded_idx, cols)

    counts = torch.zeros(H * W, dtype=cols.dtype, device=device)
    counts.scatter_add_(0, lin_idx, torch.ones_like(lin_idx, dtype=cols.dtype))

    populated = counts > 0
    if torch.any(populated):
        img_flat[populated] /= counts[populated].unsqueeze(-1)

    img = img_flat.view(H, W, 3).clamp(0.0, 1.0)
    return (img * 255).byte().cpu().numpy()


def render_from_camera(camera_position):
    """Render using the Metal-friendly torch projector."""
    print("Rendering point cloud via torch projector...")
    return torch_projective_render(means, colors, camera_position, img_width, img_height)


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
