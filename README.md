# Team 35: Tennis Match Analysis

Video analysis modules for our tennis app. They find the court, track players and
the ball, and estimate when and where each player hits the ball. Each folder is a
separate module with its own README and `requirements.txt`. They connect through
CSV/JSON files.

All models are **pretrained, used for inference only**. We do not train anything in
this repository.

## Modules

| Folder | What it does | Model | Code origin |
|---|---|---|---|
| [`court_calibration/`](court_calibration/README.md) | 14 court keypoints, homography, top-down video, per-frame image → court (metres) matrices | TennisCourtDetector | [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) + our scripts |
| [`player_tracking/`](player_tracking/README.md) | Person boxes, track IDs, 17 body keypoints, court position | YOLO11 Pose + ByteTrack | [Ultralytics](https://docs.ultralytics.com/) (pip) + our script |
| [`ball_tracking/`](ball_tracking/README.md) | Ball position per frame | TrackNet | [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) + our wrapper |
| [`hit_detection/`](hit_detection/README.md) | Hit time, hitter, hitter's court position and zones, per-player summary | Rules (no model) | Ours |

## Pipeline

```
video ─┬─► court_calibration/export_homography.py ─► court.json ─┐
       │                                                         ▼
       ├─► player_tracking/track_players.py (+ court.json) ─► players.csv ─┐
       │                                                                   ▼
       └─► ball_tracking/track_ball.py ─► ball.csv ─────► hit_detection/detect_hits.py ─► hits.csv + summary.json
```

## Quick start

Each module needs its own weights. Each weights file is **not in Git**:

- `court_calibration/weights/court_model.pt`: see the court_calibration README.
- `ball_tracking/weights/ball_model.pt`: see the ball_tracking README.
- `yolo11s-pose.pt`: downloaded automatically by Ultralytics.

Run these from the repository root, with your video at `videos/match.mp4` (also not in Git):

```bash
pip install -r court_calibration/requirements.txt -r ball_tracking/requirements.txt -r player_tracking/requirements.txt -r hit_detection/requirements.txt
```

`court_calibration/requirements.txt` has old upstream pins that may not install.
Its README explains what to do instead.

```bash
cd court_calibration && python export_homography.py --model_path weights/court_model.pt --input_path ../videos/match.mp4 --output_json ../output/court.json && cd ..
```

```bash
cd player_tracking && python track_players.py --input_path ../videos/match.mp4 --output_csv ../output/players.csv --court_homography ../output/court.json --players_only && cd ..
```

```bash
cd ball_tracking && python track_ball.py --model_path weights/ball_model.pt --input_path ../videos/match.mp4 --output_csv ../output/ball.csv --interpolate && cd ..
```

```bash
cd hit_detection && python detect_hits.py --ball_csv ../output/ball.csv --players_csv ../output/players.csv --output_csv ../output/hits.csv --summary_json ../output/summary.json && cd ..
```

All outputs use the **original video's pixel coordinates** and **frame numbers
starting at 0**, so the files line up frame by frame. Court positions are in metres.

## Status

| Feature | Status |
|---|---|
| Detect and track people, body keypoints | Implemented (`player_tracking`) |
| Filter out non-players by court position | Implemented (simple rectangle around the court) |
| Ball tracking | Implemented (`ball_tracking`) |
| Hit time + hitter + hitter's court position | Implemented as **rules** (`hit_detection`), thresholds need tuning |
| Top-down 2D court video | Implemented (`court_calibration/infer_top_down_video.py`) |
| Performance in different situations (per-player zone stats) | Basic counts in `summary.json` |
| Stable player identity across cuts / change of ends | **Not implemented**, track IDs only |
| Trajectories / heatmaps drawn on the court | **Not implemented**, data is in `players.csv` |
| Bounce detection | **Not implemented**. [yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject) has a CatBoost bounce model we could add |
| Learned hit detection (E2E-Spot) | **Not integrated**, see `hit_detection/README.md` |
| Exact racket–ball contact point, ball height or speed in 3D | **Not possible** with one camera and a ground-plane homography |
| Coaching advice ("how to improve") | **Not implemented**. It would be built on top of `summary.json` / `hits.csv` (rules or an LLM) |

## Testing status

All new scripts ran on **synthetic data**:
- Court geometry and the warp were checked against known homographies.
- TrackNet and the court model ran with randomly initialised weights.
- YOLO11 Pose ran on a still image shifted over frames.
- Hit detection ran on a hand-built rally.

**None of the modules has been tested end to end on real match footage with the
pretrained weights yet.** That is the next step.

## Credits

- Court detection model and code: [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)
- Ball tracking model and code: [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet)
- Person detection, pose and tracking: [Ultralytics YOLO11](https://docs.ultralytics.com/) with ByteTrack
- Reference projects we looked at: [yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject) and
  [abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis). No code was copied from these.

We did not train the pretrained models and do not claim them as our own. Check
each upstream project for its licensing terms. The two yastrebksv repositories don't
include a license file. Ultralytics is distributed under AGPL-3.0, which matters if
the backend is offered as a network service.
