# Hit Detection Module

Estimates **when** a player hits the ball, **which tracked person** hit it, and
**where on the court** they were standing. It uses rules on the outputs of
`ball_tracking` and `player_tracking`. It has no learned model, and the thresholds
must be tuned on real footage.

## Files

```
hit_detection/
├── README.md
├── requirements.txt
└── detect_hits.py   # ours
```

## How it works

1. Short gaps in the ball track (up to `--max_gap` frames) are filled linearly.
2. For each frame it compares the ball's movement over the previous and next
   `--window` frames. A frame is a candidate when the path turns by at least
   `--min_angle` **and** the vertical image direction reverses. Reversal means the
   ball switched between travelling towards and away from the camera. Bounces
   usually keep that direction, so this rule removes most of them. Use `--any_turn`
   to turn the reversal check off.
3. Of candidates closer than `--min_gap_s` to each other, only the sharpest turn is
   kept.
4. A candidate becomes a hit only if the ball is within `--reach` × box height of a
   person's bounding box. The nearest person is the hitter. If the player CSV has an
   `in_play_area` column, only those people are considered. `--player_ids`
   restricts the search to track IDs you chose by hand.
5. If court coordinates are available, the hitter's position is put into zones.

## Run

Run these from the repository root, after the other modules have produced their CSVs
for the same video:

```bash
cd hit_detection
pip install -r requirements.txt
```

```bash
python detect_hits.py --ball_csv ../ball_tracking/output/ball.csv --players_csv ../player_tracking/output/players.csv --output_csv output/hits.csv --summary_json output/summary.json
```

| Argument | Default | Description |
|---|---|---|
| `--ball_csv` / `--players_csv` | required | Outputs of the other modules for the same video |
| `--output_csv` | required | One row per detected hit |
| `--summary_json` | none | Per-track statistics |
| `--window` | `3` | Frames before/after used to measure the ball's direction |
| `--max_gap` | `5` | Longest ball gap (frames) that is filled |
| `--min_move` | `2.0` | Minimum ball movement in pixels over `--window` frames |
| `--min_angle` | `45` | Minimum turn angle in degrees |
| `--any_turn` | off | Don't require a vertical direction reversal |
| `--min_gap_s` | `0.4` | Minimum time between hits |
| `--reach` | `0.6` | Maximum ball-to-box distance, as a fraction of the box height |
| `--player_ids` | all | e.g. `1,4` |
| `--include_all_people` | off | Ignore `in_play_area` |

**Output CSV (`hits.csv`)**

- `frame, time_s, track_id`: when the hit happened and the track ID of the hitter.
- `ball_x, ball_y, turn_angle_deg`: ball position in the image and how sharply its
  path turned.
- `ball_to_box, ball_to_wrist`: distances divided by the hitter's box height.
- `wrist_speed`: maximum wrist movement per frame within ±2 frames, divided by box
  height. It's a rough swing indicator, recorded but not used to accept hits.
- `ground_x, ground_y`: the hitter's ground point in the image.
- `court_x_m, court_y_m, side, behind_baseline_m, depth_zone, lateral_zone`: the
  hitter's **standing position** at the hit. `side` is `far` or `near`.
  `depth_zone` is one of `behind_baseline` (≥ 1 m behind), `baseline` (within
  1 m), `mid_court` or `net` (within 4 m of the net). `lateral_zone` is the court
  width split into thirds `left`/`center`/`right`, **as seen from the camera**, not
  from the player's point of view.

`summary.json` counts hits, sides and zones per track ID, plus the average distance
behind the baseline.

## What "hit point" means here

| Meaning | Status |
|---|---|
| Time of the hit | **Estimated** (frame and timestamp) |
| Where the hitter stood | **Estimated** from the ground point + court homography |
| Exact 3D contact point / height of the ball on the racket | **Not possible** with this setup. The court homography only covers the ground plane, so a ball in the air can't be mapped to real positions or speeds with it |

## Limitations

- These are heuristics. Bounces near a player, missed detections, camera cuts and
  serves (the ball is tossed up first) can cause false or missed hits.
- Hits are attributed to **track IDs**, which are not stable player identities.
- Tested only on a synthetic rally: three hits, two bounces and one turn with no
  player nearby. All three hits were found and assigned correctly, and the bounces
  and the far turn were rejected. It has **not** been tested on real match data.

## Possible upgrade: E2E-Spot

[E2E-Spot](https://github.com/jhong93/spot) (ECCV 2022) is a video event spotting
model with tennis checkpoints (for example `tennis_rny002gsm_gru_rgb`). It outputs
event frames, classes and scores. It could replace step 2–3's rule for hit timing.
It is **not integrated**. It needs its own frame extraction and the
`config.json` that matches each checkpoint. It does not give a contact position and
does not know our track IDs.
