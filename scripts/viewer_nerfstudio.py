import argparse
import subprocess
import pathlib
import sys


def main():
    parser = argparse.ArgumentParser(
        description="Launch Nerfstudio viewer on a Gaussian splat / PLY file."
    )
    parser.add_argument(
        "--ply",
        type=str,
        default="assets/pointclouds/pilot_plant_devices.ply",
        help="Path to the .ply Gaussian splat file.",
    )
    args = parser.parse_args()

    ply_path = pathlib.Path(args.ply).expanduser().resolve()

    if not ply_path.exists():
        print(f"[ERROR] PLY file not found: {ply_path}")
        sys.exit(1)

    # Command to run nerfstudio viewer
    cmd = ["ns-viewer", "--load", str(ply_path)]

    
    print(f"Launching Nerfstudio viewer with file:\n  {ply_path}")
    print("Make sure you're inside the nsenv (virtual env).")
    

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


if __name__ == "__main__":
    main()

