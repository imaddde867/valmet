from plyfile import PlyData
import numpy as np
import viser

SH_C0 = 0.28209479177387814

# Load
ply = PlyData.read("assets/pointclouds/pilot_plant_devices.ply")
v = ply['vertex']

# XYZ
points = np.vstack([v['x'], v['y'], v['z']]).T

# Base colors
f_dc = np.vstack([v['f_dc_0'], v['f_dc_1'], v['f_dc_2']]).T

colors = (SH_C0 * f_dc + 0.5)
colors = np.clip(colors, 0.0, 1.0)
colors = (colors * 255).astype(np.uint8)

server = viser.ViserServer()

server.scene.add_point_cloud(
    "/gaussians",
    points=points,
    colors=colors,
    point_size=0.01,
)


print("Viewer running at http://localhost:8080")

import time
while True:
    time.sleep(1)
