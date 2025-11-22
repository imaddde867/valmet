#!/usr/bin/env python3
"""
Simple PLY viewer using Open3D.
This can view point clouds and meshes directly without needing a trained model.
"""

import argparse
import pathlib
import sys

try:
    import open3d as o3d
    import numpy as np
except ImportError:
    print("[ERROR] Required packages not found. Make sure you're in the nsenv:")
    print("  source nsenv/bin/activate")
    sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="View PLY files (point clouds or meshes) using Open3D."
    )
    parser.add_argument(
        "ply_file",
        type=str,
        help="Path to the .ply file to visualize.",
    )
    parser.add_argument(
        "--point-size",
        type=float,
        default=1.0,
        help="Point size for rendering (default: 1.0).",
    )
    parser.add_argument(
        "--background",
        type=str,
        default="black",
        choices=["black", "white", "gray"],
        help="Background color (default: black).",
    )
    args = parser.parse_args()

    ply_path = pathlib.Path(args.ply_file).expanduser().resolve()

    if not ply_path.exists():
        print(f"[ERROR] PLY file not found: {ply_path}")
        sys.exit(1)

    print(f"[INFO] Loading PLY file: {ply_path}")

    try:
        # Try loading as a point cloud first
        geometry = o3d.io.read_point_cloud(str(ply_path))
        
        if not geometry.has_points():
            print("[INFO] Not a point cloud, trying as triangle mesh...")
            geometry = o3d.io.read_triangle_mesh(str(ply_path))
            
            if not geometry.has_triangles():
                print("[ERROR] File doesn't contain valid point cloud or mesh data.")
                sys.exit(1)
            
            geometry_type = "mesh"
            print(f"[INFO] Loaded mesh with {len(geometry.triangles)} triangles")
            
            # Compute normals for better visualization
            if not geometry.has_vertex_normals():
                geometry.compute_vertex_normals()
        else:
            geometry_type = "point_cloud"
            print(f"[INFO] Loaded point cloud with {len(geometry.points)} points")
            
            # Estimate normals if not present
            if not geometry.has_normals():
                print("[INFO] Computing normals...")
                geometry.estimate_normals(
                    search_param=o3d.geometry.KDTreeSearchParamHybrid(
                        radius=0.1, max_nn=30
                    )
                )

        # Set up visualization
        vis = o3d.visualization.Visualizer()
        vis.create_window(window_name=f"PLY Viewer - {ply_path.name}")
        vis.add_geometry(geometry)
        
        # Configure rendering options
        opt = vis.get_render_option()
        if geometry_type == "point_cloud":
            opt.point_size = args.point_size
        
        # Set background color
        bg_colors = {
            "black": [0.0, 0.0, 0.0],
            "white": [1.0, 1.0, 1.0],
            "gray": [0.5, 0.5, 0.5]
        }
        opt.background_color = np.array(bg_colors[args.background])
        
        # Set up nice default view
        view_control = vis.get_view_control()
        view_control.set_zoom(0.8)
        
        print("\n" + "="*60)
        print("VIEWER CONTROLS:")
        print("="*60)
        print("  Mouse Left:   Rotate")
        print("  Mouse Right:  Pan")
        print("  Mouse Wheel:  Zoom")
        print("  Q / Esc:      Quit")
        print("  H:            Show help")
        print("="*60 + "\n")
        
        # Run visualization
        vis.run()
        vis.destroy_window()
        
        print("[INFO] Viewer closed.")
        
    except Exception as e:
        print(f"[ERROR] Failed to load or visualize PLY file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()