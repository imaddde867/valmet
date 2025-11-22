# some quick command shit

- `pip install viser numpy plyfile` - install the Python dependencies used by the tooling scripts.
- `python3 -m http.server 8080 --directory web` - serve `web/index.html` locally at http://localhost:8080.
- `python scripts/frame_sampler.py --input-video assets/videos/Pilot_plant.mp4 --frame-interval 12 --frames-dir outputs/frames/model --manifest outputs/frame_manifest.json --annotation-dir outputs/frames/annotation --annotation-max-dim 3840 --annotation-target-megapixels 6 --annotation-format jpg --annotation-jpeg-quality 98` - export TFJS-sized frames plus high-res annotation frames in one pass.
- `python scripts/tfjs_inference.py --manifest outputs/frame_manifest.json --frames-dir outputs/frames/model --output outputs/detections/pilot_plant.json` - run the TFJS baseline against the freshly sampled frames (per-class thresholds configurable via `--thresholds config/thresholds.json`).
