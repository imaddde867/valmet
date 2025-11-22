# some quick command shit

- `pip install -r requirements.txt` - install all Python dependencies for the sampler, COLMAP helpers, etc.
- `python3 -m http.server 8080` - serve the repo root locally (visit http://localhost:8080/web/ so `web/main.js` can fetch `../assets` and `../outputs`).
- `python scripts/frame_sampler.py --input-video assets/videos/Pilot_plant.mp4 --frame-interval 12 --frames-dir outputs/frames/highres --manifest outputs/frame_manifest.json --frames-format jpg --frames-jpeg-quality 95` - export a single set of high-resolution frames (defaults keep aspect ratio and cap at 4K).
- `python scripts/tfjs_inference.py --manifest outputs/frame_manifest.json --frames-dir outputs/frames/highres --output outputs/detections/pilot_plant.json --input-width 224 --input-height 224` - run the TFJS baseline; frames are resized inside the script so accuracy stays tied to the HQ samples (per-class thresholds configurable via `--thresholds config/thresholds.json`).
