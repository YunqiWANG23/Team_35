# Team 35: Tennis Serve Analysis

Our goal is to help players **correct their serve contact point**. The system has
two parts:

1. **Recognition.** Detect and track the players and the ball, find hits, and
   recognise events (serve / overhead / groundstroke). For each hit it estimates
   the time, the hitter's court position and the contact height.
2. **Suggestion.** For every serve event, simulate how the serve would change with a
   different launch angle, aim, racquet speed, spin (+top / −back), contact height or
   contact position. The output says which changes make the serve more forgiving.

Each folder is a separate module with its own README and `requirements.txt`. The
modules connect through CSV/JSON files, and `run_pipeline.py` runs them all on one
video. The detection models are **pretrained and only used for inference**. We do not
train anything in this repository. The suggestion part is a **physics simulation**,
not a trained model.

## Modules

| Part | Folder | What it does | Model / method | Code origin |
|---|---|---|---|---|
| 1 | [`court_calibration/`](court_calibration/README.md) | Court keypoints, homography, top-down video, per-frame image → court (metres) matrices | TennisCourtDetector | [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector) + our scripts |
| 1 | [`player_tracking/`](player_tracking/README.md) | Person boxes, track IDs, 17 body keypoints, court position | YOLO11 Pose + ByteTrack | [Ultralytics](https://docs.ultralytics.com/) (pip) + our script |
| 1 | [`ball_tracking/`](ball_tracking/README.md) | Ball position per frame | TrackNet | [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet) + our wrapper |
| 1 | [`hit_detection/`](hit_detection/README.md) | Hit time and hitter, court zones, event type, contact height estimate | Rules | Ours |
| 2 | [`serve_simulation/`](serve_simulation/README.md) | What-if serve simulations, launch-angle windows, suggestions, plots, before/after JSON | Drag + Magnus physics (RK4) | Flight model ported from [orcn-dev/tennis-physics-3d-demo](https://github.com/orcn-dev/tennis-physics-3d-demo). Parameters follow [TopSpin Ball Lab](https://topspin-labs.vercel.app/ball-lab) |

## Pipeline

```
                    ┌─ court_calibration/export_homography.py ─► court.json ─┐
                    │                                                        ▼
uploaded video ─────┼─ player_tracking/track_players.py (+court.json) ─► players.csv ─┐
                    │                                                                ▼
                    └─ ball_tracking/track_ball.py ─► ball.csv ──► hit_detection/detect_hits.py ─► hits.csv
                                                                                                    │
                       hit_detection/classify_events.py ◄──────────────────────────────────────────┘
                                   │ events.csv
                                   ▼
                       serve_simulation/simulate_serves.py ─► serves/serve_<frame>.json + .png
                                                              serves/serves_summary.json
```

## Quick start

1. Get the weights. Each weights file is **not in Git**:
   - `court_calibration/weights/court_model.pt`: see the court_calibration README.
   - `ball_tracking/weights/ball_model.pt`: see the ball_tracking README.
   - `yolo11s-pose.pt`: downloaded automatically by Ultralytics.
2. Install the dependencies (from the repository root):

```bash
pip install -r court_calibration/requirements.txt -r ball_tracking/requirements.txt -r player_tracking/requirements.txt -r hit_detection/requirements.txt -r serve_simulation/requirements.txt
```

`court_calibration/requirements.txt` has old upstream pins that may not install.
Its README explains what to do instead.

3. Run the whole pipeline on a video (also not in Git):

```bash
python run_pipeline.py --video videos/match.mp4 --output_dir output/match --court_model court_calibration/weights/court_model.pt --ball_model ball_tracking/weights/ball_model.pt
```

Useful options:
- `--device cpu|cuda|mps`;
- `--racquet_speed` and `--spin_rpm`, the assumptions used by the simulation;
- `--player_height`, used for the contact height estimate;
- `--skip_existing`, which skips steps whose output already exists. Use it to
  re-run only the later steps.

All files use the **original video's pixel coordinates** and **frame numbers
starting at 0**. Court positions are in metres:
- x runs 0 to 10.97 m across the court;
- y runs 0 m (far baseline) to 23.77 m (near baseline).

### Backend integration (the "adjust this serve" panel)

1. After upload, run `run_pipeline.py`.
2. Show the events from `events.csv` on a timeline, using `frame`/`time_s` to jump
   to the clip.
3. When the user picks a serve, show `serves/serve_<frame>.json`: its baseline,
   suggestions and plot.
4. When the user moves a slider, call `serve_simulation/whatif.py` with the serve's
   server position and the changed parameters. It returns before/after outcomes and
   trajectories to draw.
5. Always label these results **"Simulation"**. Racquet speed, spin and launch angle
   are assumptions, not measurements from the video.

## Status

| Feature | Status |
|---|---|
| Detect and track people, body keypoints | Implemented (`player_tracking`) |
| Ball tracking | Implemented (`ball_tracking`) |
| Hit time + hitter + court position | Implemented as **rules** (`hit_detection`), thresholds need tuning |
| Event recognition: serve / overhead / groundstroke | Implemented as **rules** (`classify_events.py`) |
| Contact height | **Rough estimate** from pose scale |
| Serve what-if simulation (angle, aim, speed, spin, contact height/position) | Implemented (`serve_simulation`), physics not calibrated against measured serves |
| Suggestions | Rule-based text over simulation results, labelled "Simulation" |
| Top-down 2D court video | Implemented (`court_calibration/infer_top_down_video.py`) |
| Measuring racquet speed, spin or launch angle from the video | **Not possible** with the current single-camera 2D tracking. These are inputs/assumptions |
| Slice / kick serves (sidespin), after-bounce behaviour | **Not modelled** |
| Bounce detection, learned event spotting (E2E-Spot) | **Not integrated** |
| Stable player identity across cuts / change of ends | **Not implemented**, track IDs only |
| Personalised coaching beyond the simulation rules | **Not implemented** |

## Testing status

All new code was run on **synthetic data**:
- Court geometry and the metre conversion were checked against known homographies.
- TrackNet and the court model ran with random weights.
- YOLO11 Pose ran on a still image shifted over frames.
- Hits and events were checked on hand-built rallies with serves. Serves from both
  sides were found at the right frames.
- The physics was checked against the analytic projectile (no drag).
- `run_pipeline.py` ran end to end on a test video.

**Nothing has been tested on real match footage with the pretrained weights yet.**
That is the next step, followed by tuning the hit and serve thresholds.

## Credits

- Court detection model and code: [yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)
- Ball tracking model and code: [yastrebksv/TrackNet](https://github.com/yastrebksv/TrackNet)
- Person detection, pose and tracking: [Ultralytics YOLO11](https://docs.ultralytics.com/) with ByteTrack
- Ball flight model (ported): [orcn-dev/tennis-physics-3d-demo](https://github.com/orcn-dev/tennis-physics-3d-demo)
- Serve parameter set inspired by [TopSpin Ball Lab](https://topspin-labs.vercel.app/ball-lab). No code, model or API is used from it.
- Reference projects we looked at: [yastrebksv/TennisProject](https://github.com/yastrebksv/TennisProject) and
  [abdullahtarek/tennis_analysis](https://github.com/abdullahtarek/tennis_analysis). No code was copied from these.

We did not train the pretrained models and do not claim them as our own. Check
each upstream project for its licensing terms:
- The yastrebksv repositories and tennis-physics-3d-demo don't include a license
  file.
- Ultralytics is distributed under AGPL-3.0, which matters if the backend is
  offered as a network service.
