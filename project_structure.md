# Project Structure

- `docs/` – Planning and challenge material (PLAN, brief PDFs, PPT).
- `assets/` – Source data for the demo.
  - `assets/pointclouds/` – Gaussian splat PLY scenes.
  - `assets/meshes/` – GLB conversions of the same scenes.
  - `assets/videos/` – Capture videos used for detection + alignment.
  - `assets/images/` – Still imagery or reference frames (e.g., Devices.png).
- `scripts/` – Python utilities and automation (e.g., `viewer_viser.py` for quick PLY inspection).
- `web/` – Front-end artifacts and TFJS models for the baseline demo.
  - `web/models/tfjs_baseline/` – AutoML TFJS detector weights and config.
- `outputs/` – Placeholder for generated detections, tracks, and metrics (`.gitkeep` to retain the directory).
- `archive/` – (Currently empty) Reserved for any future orphaned files pending review.
- `readme.md` – Top-level setup instructions.
- `venv/` – Local Python virtual environment (left untouched).
