# HSV Tuner WebUI (Contour/HSV only)

This folder contains a standalone HSV tuning tool that reuses the existing contour detector code from the repo.

## What it does
- Input source: image, AVI/video, or webcam.
- Stereo side-by-side split: left/right/both.
- Detection modes: balloon, goal, all, custom.
- Object registry with presets (green/purple balloons, orange/yellow goals) plus dynamic custom objects.
- Per-object tuning UI: select one object at a time, add/duplicate/remove custom objects, tune HSV and thresholds.
- Mode-aware source input:
  - `image` / `video`: choose file with native picker or drag-and-drop upload
  - `camera`: quick preset (`webcam=0` / `usb=1`) with manual index fallback
- Live panes: detection view + mask with multi-bboxes.
- Pixel sampling: click detection view to inspect BGR/HSV.
- Debug console panel with runtime status and detections.
- Save tuning profiles (`profiles/*.json`) and import profile JSON files.
- Export ROS2 YAML (`exports/*.yaml`) and import YAML files.
- Profile/YAML import supports native file picker and drag-and-drop.
- Save snapshots (`snapshots/*.jpg`).

## Run
From repository root:

```bash
python3 -m pip install -r requirements.txt
python3 main.py --host 127.0.0.1 --port 8765
```

Then open:

- http://127.0.0.1:8765

## Notes
- This is for HSV/contour tuning only (not RKNN/YOLO runtime parity).
- Detector code requires `blimp_vision` to be importable. Preferred options:
  - keep this monorepo layout (`blimp_vision/` at repo root), or
  - install `blimp_vision` as a dependency (submodule/vendor/pip), or
  - set `BLIMP_VISION_PATH=/path/to/repo_containing_blimp_vision`.
- ROS2 YAML output uses `vision_tuning.ros__parameters` with a modular tree:
  - `detectors.green_balloon.hsv.*`, `detectors.goal_orange.hsv.*`, etc.
  - `tracking.use_kalman`, `tracking.use_optical_flow`, `tracking.use_lock`
- Legacy flat keys are still exported for backward compatibility with older nodes.
