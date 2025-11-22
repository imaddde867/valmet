# Frame Sampling & Annotation Quality

The plan in `docs/PLAN.md` calls for precise detection boxes that ultimately feed
tracking, triangulation, and the polished TFJS overlay. That workflow only stays
accurate when the intermediate frames we review and annotate retain enough pixel
density to resolve small valves, positioners, and similar industrial parts.

The upgraded `scripts/frame_sampler.py` script now emits both
model-friendly frames and optional annotation-grade frames so we can balance
resolution, file size, and downstream inference requirements without juggling
multiple utilities.

## Recommended workflow

1. Decide on the sampling cadence (`--frame-interval`) that keeps temporal
   coverage tight enough for tracking (for Pilot_plant we use 10–15 frames).
2. Export the model frames at the detector’s native input size. If the TFJS
   model expects 224×224 inputs, use the defaults or enable `--keep-aspect`
   plus padding downstream.
3. Enable annotation exports to maintain visual fidelity when drawing boxes.
   Example command:

   ```bash
   python scripts/frame_sampler.py \
     --input-video assets/videos/Pilot_plant.mp4 \
     --frame-interval 12 \
     --frames-dir outputs/frames/model \
     --manifest outputs/frame_manifest.json \
     --annotation-dir outputs/frames/annotation \
     --annotation-max-dim 3840 \
     --annotation-target-megapixels 6 \
     --annotation-format jpg \
     --annotation-jpeg-quality 98
   ```

   This keeps inference images lightweight (224×224 PNG by default) while also
   writing `frame_*_hq.jpg` files near the original resolution but capped at
   6 MP and 3,840 px on the long edge, which matches the typical crop that
   keeps handheld capture sharp on 4K monitors.

4. Review bounding boxes inside any labeling tool that points to the
   `annotation` variant recorded in the manifest (`variants.annotation`). The
   JSON includes per-variant scale factors so detections can be lifted back to
   the higher-resolution canvases when needed.

## Option reference

- `--keep-aspect` and `--interpolation`: make the model-resize step deterministic
  if we ever ship a non-square backbone or need Lanczos downsampling.
- `--frames-format` / JPEG/PNG quality knobs: trade off storage vs. fidelity
  for inference frames when running ablations on edge devices with tight IO.
- `--annotation-max-dim` limits the longest edge; use it to keep exports inside
  a chosen monitor resolution (e.g., 2160 for UHD).
- `--annotation-target-megapixels` enforces a pixel budget, which approximates
  PPI when we assume a 27″ QA monitor (~110 PPI). Setting 6 MP keeps the per-inch
  granularity high enough for small gauges without exploding disk usage.

These knobs give us precise control over the trade space between resolution,
pixel density, and file size so we can quickly annotate for the hackathon while
still feeding a consistent manifest into `scripts/tfjs_inference.py`.
