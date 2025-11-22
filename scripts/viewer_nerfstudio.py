#!/usr/bin/env python3
"""
Launch Nerfstudio viewer for visualizing Gaussian splat PLY files or trained NeRF models.
"""

import argparse
import subprocess
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Launch Nerfstudio viewer on a Gaussian splat / PLY file or config."
    )
    parser.add_argument(
        "--ply",
        type=str,
        help="Path to the .ply Gaussian splat file (for direct PLY viewing).",
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to a Nerfstudio config.yml file (for trained models).",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7007,
        help="Websocket port for the viewer (default: 7007).",
    )
    args = parser.parse_args()

    # Check that at least one input is provided
    if not args.ply and not args.config:
        print("[ERROR] You must provide either --ply or --config")
        parser.print_help()
        sys.exit(1)

    # If a PLY file is provided, we need to use it differently
    # Note: ns-viewer expects a config file, not a raw PLY
    # For PLY files, you might need to use gsplat or another viewer
    if args.ply:
        ply_path = pathlib.Path(args.ply).expanduser().resolve()
        
        if not ply_path.exists():
            print(f"[ERROR] PLY file not found: {ply_path}")
            sys.exit(1)
        
        print(f"[INFO] PLY file found: {ply_path}")
        print("[INFO] Note: ns-viewer expects a config file from a trained model.")
        print("[INFO] For viewing raw PLY files, you may need to use alternative tools.")
        print("[INFO] If this is a Gaussian splat PLY, consider using:")
        print("       - Open3D: python -c 'import open3d as o3d; pcd = o3d.io.read_point_cloud(\"path.ply\"); o3d.visualization.draw_geometries([pcd])'")
        print("       - Or train a model first with: ns-train splatfacto --data /path/to/data")
        sys.exit(1)

    # If a config file is provided
    if args.config:
        config_path = pathlib.Path(args.config).expanduser().resolve()
        
        if not config_path.exists():
            print(f"[ERROR] Config file not found: {config_path}")
            sys.exit(1)

        # Command to run nerfstudio viewer with config
        cmd = [
            "ns-viewer",
            "--load-config", str(config_path),
            "--viewer.websocket-port", str(args.port)
        ]
        
        print(f"[INFO] Launching Nerfstudio viewer with config:\n  {config_path}")
        print(f"[INFO] Viewer will be available at: http://localhost:{args.port}")
        print("[INFO] Make sure you're inside the nsenv virtual environment.")
        
        try:
            subprocess.run(cmd, check=True)
        except FileNotFoundError:
            print(
                "[ERROR] 'ns-viewer' not found.\n"
                "Make sure Nerfstudio is installed in this venv and that you're in nsenv:\n"
                "  source nsenv/bin/activate\n"
                "and that 'ns-viewer' runs from the terminal."
            )
            sys.exit(1)
        except subprocess.CalledProcessError as e:
            print(f"[ERROR] ns-viewer exited with error code {e.returncode}")
            sys.exit(e.returncode)
        except KeyboardInterrupt:
            print("\n[INFO] Viewer stopped by user.")
            sys.exit(0)


if __name__ == "__main__":
    main()