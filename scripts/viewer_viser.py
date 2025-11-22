import time
import numpy as np
import torch
import viser
from plyfile import PlyData

from device_utils import VALMET_DEVICE_ENV, pick_device

# Read the PLY file
plydata = PlyData.read("assets/pointclouds/pilot_plant_devices.ply")
vertex = plydata['vertex']

# Select compute device (Metal / CUDA / CPU)
device = pick_device()
print(f"Preparing point cloud on device: {device} (override via {VALMET_DEVICE_ENV})")

# Extract positions
positions_np = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

# Extract colors from spherical harmonics DC component and convert on device
SH_C0 = 0.28209479177387814
colors_sh_np = np.vstack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']]).T

positions = torch.from_numpy(positions_np).float().to(device)
colors_sh = torch.from_numpy(colors_sh_np).float().to(device)
colors_rgb = torch.clamp(colors_sh * SH_C0 + 0.5, 0.0, 1.0)

positions = positions.cpu().numpy()
colors_rgb = (colors_rgb * 255).byte().cpu().numpy()

# Create viser server
server = viser.ViserServer()
server.scene.add_point_cloud(
    "/gaussians",
    points=positions,
    colors=colors_rgb,
    point_size=0.01
)

print("Open http://localhost:8080 in your browser")
while True:
    time.sleep(1)
