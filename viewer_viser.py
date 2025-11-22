import viser
import numpy as np
from plyfile import PlyData

# Lire le fichier PLY
plydata = PlyData.read("PLY files/pilot_plant_devices.ply")
vertex = plydata['vertex']

# Extraire les positions
positions = np.vstack([vertex['x'], vertex['y'], vertex['z']]).T

# Créer le serveur viser
server = viser.ViserServer()
server.scene.add_point_cloud(
    "/gaussians",
    points=positions,
    colors=np.random.randint(0, 255, (len(positions), 3)),  # couleurs aléatoires
    point_size=0.01
)

print("Ouvre http://localhost:8080 dans ton navigateur")
while True:
    pass
