# Player Tracking Module

Detects and tracks people with **YOLO11 Pose** and the **ByteTrack** tracker, both
from [Ultralytics](https://docs.ultralytics.com/tasks/pose/). For each person in
each frame it outputs a bounding box, a track ID and 17 body keypoints. If a court
homography is available, it also outputs the person's court position in metres.

YOLO11 Pose is a general person/pose model. It is not trained specifically on tennis
and it does not judge technique. No Ultralytics code is copied into this repository.
It is installed with pip.

## Files

```
player_tracking/
├── README.md
├── requirements.txt
└── track_players.py   # ours
```

## 1. Weights

Nothing to download by hand. Ultralytics downloads `yolo11s-pose.pt` automatically
on first run, into the folder you run the command from. `.gitignore` excludes
`*.pt`, so the file won't be committed. Use `--model yolo11s.pt` if you don't need
poses, or `yolo11n-pose.pt` / `yolo11m-pose.pt` for a smaller or larger model.

## 2. Install and run

Run these from the repository root:

```bash
cd player_tracking
pip install -r requirements.txt
```

Basic run:

```bash
python track_players.py --input_path ../videos/match.mp4 --output_csv output/players.csv --output_video output/players.mp4
```

With court coordinates and only people in the play area. This needs a JSON file from
`court_calibration/export_homography.py` for the **same video**:

```bash
python track_players.py --input_path ../videos/match.mp4 --output_csv output/players.csv --court_homography ../court_calibration/output/court.json --players_only
```

| Argument | Default | Description |
|---|---|---|
| `--input_path` | required | Input video |
| `--output_csv` | required | One row per tracked person per frame |
| `--model` | `yolo11s-pose.pt` | Ultralytics model name or path |
| `--tracker` | `bytetrack.yaml` | Ultralytics tracker config |
| `--conf` | `0.25` | Detection confidence threshold |
| `--imgsz` | `640` | Inference image size. Players far from the camera are small, so a larger value (e.g. `1280`) may help, at the cost of speed |
| `--court_homography` | none | Per-frame image → court matrices |
| `--margin_side` / `--margin_end` | `3.0` / `6.0` | Metres beyond the sidelines / baselines still counted as the play area |
| `--players_only` | off | Drops people outside the play area (umpire, ball kids, crowd). Needs `--court_homography` |
| `--output_video` | none | Optional annotated video from Ultralytics |
| `--device` | auto | e.g. `cpu`, `0` (first CUDA GPU), `mps` |

**Output CSV columns**

- `frame, track_id, conf, x1, y1, x2, y2`: box in original video pixels.
- `ground_x, ground_y, ground_source`: where the person stands. This is the ankle
  midpoint when both ankles are confident (`ankles`), otherwise the bottom centre of
  the box (`box_bottom`).
- `court_x_m, court_y_m, in_play_area`: only filled with `--court_homography`. The
  court frame comes from `export_homography.py`: x runs 0 to 10.97 m across the
  court and y runs 0 m (far baseline) to 23.77 m (near baseline).
- `<keypoint>_x, <keypoint>_y, <keypoint>_conf` for the 17 COCO keypoints (`nose` …
  `right_ankle`).

## Limitations

- **A track ID is not a player identity.** IDs can change after occlusion, camera
  cuts or a change of ends. Check them, or pick the player IDs by hand (see
  `hit_detection --player_ids`).
- The play-area filter is a simple rectangle in court metres. Line judges or ball
  kids standing inside it are kept.
- Court coordinates only make sense for points **on the ground**. The ground point
  is an approximation, especially when the feet are hidden.
- Each run processes one video with its own tracker state. A backend must not share
  a tracker between parallel jobs.
- Tested on a synthetic clip (a still image shifted over 20 frames) with the real
  `yolo11s-pose.pt`: 5 people were tracked with stable IDs, and the ankle ground
  points and court mapping worked. It has **not** been tested on real tennis footage.
