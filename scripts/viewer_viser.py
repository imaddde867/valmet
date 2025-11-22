import viser
import numpy as np
from plyfile import PlyData

# Read the PLY file
plydata = PlyData.read("assets/pointclouds/pilot_plant_devices.ply")
vertex = plydata['vertex']

# Extract positions
positions = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

# Extract colors from spherical harmonics DC component
# f_dc_0, f_dc_1, f_dc_2 are the base colors in SH space
# Convert from SH to RGB (SH_C0 = 0.28209479177387814)
SH_C0 = 0.28209479177387814
colors_sh = np.vstack([vertex['f_dc_0'], vertex['f_dc_1'], vertex['f_dc_2']]).T
colors_rgb = (colors_sh * SH_C0 + 0.5) * 255
colors_rgb = np.clip(colors_rgb, 0, 255).astype(np.uint8)

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
    pass