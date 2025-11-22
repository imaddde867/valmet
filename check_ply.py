from plyfile import PlyData
ply = PlyData.read("assets/pointclouds/pilot_plant_devices.ply")
print(ply['vertex'].data.dtype.names)
