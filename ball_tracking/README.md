# Ball Tracking Module

Tracks the tennis ball in broadcast-style videos with **TrackNet**. The model and
code come from [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet). It
takes three consecutive frames as input and predicts a heatmap of the ball position.
We only run inference with the author's pretrained weights, no training.

This module tracks **the ball only**, not players. It outputs the ball's position in
the image for each frame. It does not detect hits or bounces and does not give real
3D positions or speeds.

## Files

```
ball_tracking/
├── README.md
├── SOURCE.md          # attribution and the upstream commit
├── requirements.txt
├── model.py           # upstream: BallTrackerNet (9 input channels, 256 output classes)
├── general.py         # upstream: heatmap postprocess (+ training helpers, unused here)
├── infer_on_video.py  # upstream: original inference script and track cleaning functions
├── track_ball.py      # ours: streaming wrapper that writes a CSV
└── weights/           # NOT in Git: put ball_model.pt here
```

The upstream files are copied unchanged.

## 1. Weights

Download the pretrained TrackNet weights:

https://drive.google.com/file/d/1XEYZ4myUN7QT-NeBYJI0xteLsvs-ZAOl/view

Save them as `ball_tracking/weights/ball_model.pt`. This is a PyTorch `state_dict`
for `model.BallTrackerNet`. It **cannot** be loaded with `ultralytics.YOLO`, and it
is a different network from the court model in `court_calibration`. You don't need
the training dataset.

## 2. Install and run

Run these from the repository root:

```bash
cd ball_tracking
pip install -r requirements.txt
```

```bash
python track_ball.py --model_path weights/ball_model.pt --input_path ../videos/match.mp4 --output_csv output/ball.csv --output_video output/ball.mp4 --interpolate
```

| Argument | Default | Description |
|---|---|---|
| `--model_path` | required | TrackNet weights |
| `--input_path` | required | Input video |
| `--output_csv` | required | Ball positions per frame |
| `--output_video` | none | Optional video with the last 7 ball positions drawn |
| `--interpolate` | off | Fills short gaps between detections (the upstream `--extrapolation` option). Gaps before the first or after the last detection are left empty |
| `--device` | `auto` | `auto` (CUDA if available, else CPU), `cpu`, `cuda`, `mps` |

**Output CSV:** `frame, time_s, x, y`. Coordinates are pixels in the **original
video**. `x` and `y` are empty when no ball was found. The first two frames are
always empty because the model needs two earlier frames.

## Notes

- Compared with upstream `infer_on_video.py`, `track_ball.py` reads frames one at a
  time, lets you choose the device (upstream always uses CUDA), and writes a CSV.
  It reuses the upstream outlier removal and interpolation functions.
- Frames are resized to 640×360 for the model. The upstream postprocessing maps
  points to 1280×720, and we rescale them to the real video size. The model was
  trained on 1280×720 broadcast footage, and other camera setups may perform worse.
- Tested only with **randomly initialised** weights on a synthetic video to check
  loading, streaming and output. It has **not** been tested with the pretrained
  weights on real footage.

## Source

Model and upstream code: [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet)
by yastrebksv, an unofficial implementation of the TrackNet paper
(https://arxiv.org/abs/1907.03698). We did not train this model. The upstream
repository does not include a license file, so check with the author before
redistributing.
