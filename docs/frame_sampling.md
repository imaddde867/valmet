# High-Resolution Frame Sampling

The plan in `docs/PLAN.md` demands crisp bounding boxes that survive tracking,
triangulation, and the future “Ask Your Plant” assistant. To remove the low-res
detour entirely, `scripts/frame_sampler.py` now emits a single high-resolution
frame set that preserves the original video fidelity (capped at a configurable
ceiling such as 4K). The TFJS inference script handles any downscaling on the
fly, so every downstream tool—LabelMe, Roboflow, COLMAP—works from the same
HQ frames.

## Recommended workflow

1. Decide on the sampling cadence (`--frame-interval`) that keeps temporal
   coverage tight enough for tracking (Pilot_plant works well with 10–12).
2. Export the frames once at high resolution:

   ```bash
   python scripts/frame_sampler.py \
     --input-video assets/videos/Pilot_plant.mp4 \
     --frame-interval 12 \
     --frames-dir outputs/frames/highres \
     --manifest outputs/frame_manifest.json \
     --frames-format jpg \
     --frames-jpeg-quality 95
   ```

   Defaults already cap the long edge at 3,840 px, lock aspect ratio, and avoid
   upscaling. Override `--resize` if you need a different ceiling.
3. Point your labeling tool directly at `outputs/frames/highres/`. The manifest
   records the original resolution for each frame so detections remain
   traceable.
4. Run `python scripts/tfjs_inference.py --frames-dir outputs/frames/highres ...`
   to generate detections. The script now resizes to the TFJS input size
   internally, so there is no separate “model” frame variant to manage.

## Option reference

- `--keep-aspect` (default) ensures we never distort the source frames.
- `--frames-format` / quality knobs still trade off storage vs. fidelity if you
  need lighter artifacts, but the intent is to keep as much detail as possible.
- Annotation-specific flags remain available if you ever need an additional
  copy (e.g., capped at 6 MP), but the canonical pipeline uses only the
  high-resolution directory.

With this setup every consumer—detectors, trackers, COLMAP, and the overlay—works
from the exact same image set, eliminating the quality mismatch that held the
project back.
