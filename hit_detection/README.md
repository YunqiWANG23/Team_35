# Hit and Event Detection Module

This module covers the recognition steps that come after tracking. It estimates:

- **when** a player hits the ball;
- **which tracked person** hit it;
- **where on the court** they were standing;
- **which event** it was: `serve`, `overhead` or `groundstroke`;
- a rough **contact height**.

It works with rules on the outputs of `ball_tracking` and `player_tracking`. There
is no learned model, and the thresholds must be tuned on real footage.

## Files

```
hit_detection/
├── README.md
├── requirements.txt
├── detect_hits.py       # ours: hit time, hitter, court zones
└── classify_events.py   # ours: serve / overhead / groundstroke + contact height
```

## How hits are detected (`detect_hits.py`)

1. Short gaps in the ball track (up to `--max_gap` frames) are filled linearly.
2. For each frame, the ball's movement over the previous and next `--window` frames
   is compared. A frame becomes a **candidate** when either trigger fires:
   - **`turn`:** the path turns by at least `--min_angle` and the vertical image
     direction reverses. Bounces usually keep their direction, so this removes most
     of them. Use `--any_turn` to turn the reversal check off.
   - **`speedup`:** the ball suddenly moves `--min_speedup` times faster. This
     catches serves, where a slow toss is followed by a fast ball. A serve from the
     far side often has no vertical reversal.
3. Each candidate needs a person within `--reach` × box height of the ball, and the
   ball must leave at least `--min_out_speed` box heights per frame. The second
   check rejects the top of the serve toss: the ball turns there, but slowly.
   People outside `in_play_area`, or not in `--player_ids`, are ignored.
4. Of the remaining candidates closer than `--min_gap_s` to each other, only the
   strongest is kept.
5. If court coordinates are available, the hitter's position is put into zones.

## How events are classified (`classify_events.py`)

| Check | Rule |
|---|---|
| `arm_raised` | A confident wrist is above the nose |
| `ball_above_head` | The ball is above the nose in the image |
| `first_in_rally` | No other hit in the previous `--rally_gap_s` (3 s) |
| **serve** | `arm_raised`, `ball_above_head` and `first_in_rally`, and the hitter is no more than `--max_inside_m` (1 m) inside the baseline |
| **overhead** | `arm_raised` and `ball_above_head`, but not a serve |
| **groundstroke** | Everything else, including volleys |

`contact_height_est_m` converts the vertical pixel distance between the ground
point and the ball into metres. It uses the ground-to-shoulder distance as a scale,
assuming shoulders are at `--shoulder_ratio` (0.82) × `--player_height` (1.80 m).
This is a **rough single-camera estimate**: it ignores perspective and how far the
ball is in front of the player. Set `--player_height` per player if you know it.

## Run

Run these from the repository root, after the other modules have produced their CSVs
for the same video:

```bash
cd hit_detection
pip install -r requirements.txt
```

```bash
python detect_hits.py --ball_csv ../output/ball.csv --players_csv ../output/players.csv --output_csv ../output/hits.csv --summary_json ../output/hits_summary.json
```

```bash
python classify_events.py --hits_csv ../output/hits.csv --players_csv ../output/players.csv --output_csv ../output/events.csv
```

`detect_hits.py` arguments:

| Argument | Default | Description |
|---|---|---|
| `--ball_csv` / `--players_csv` | required | Outputs of the other modules for the same video |
| `--output_csv` | required | One row per detected hit |
| `--summary_json` | none | Per-track statistics |
| `--window` | `3` | Frames before/after used to measure the ball's direction |
| `--max_gap` | `5` | Longest ball gap (frames) that is filled |
| `--min_move` | `2.0` | Minimum ball movement in pixels over `--window` frames |
| `--min_angle` | `45` | Minimum turn angle in degrees for the `turn` trigger |
| `--any_turn` | off | Don't require a vertical direction reversal |
| `--min_speedup` | `2.5` | Speed-up factor for the `speedup` trigger. `0` turns it off |
| `--min_out_speed` | `0.02` | Minimum outgoing ball speed, in hitter box heights per frame |
| `--min_gap_s` | `0.4` | Minimum time between hits |
| `--reach` | `0.6` | Maximum ball-to-box distance, as a fraction of the box height |
| `--player_ids` | all | e.g. `1,4` (manual player selection) |
| `--include_all_people` | off | Ignore `in_play_area` |

**`hits.csv` columns**

- `frame, time_s, track_id`: when the hit happened and the track ID of the hitter.
- `ball_x, ball_y`: ball position in the image.
- `trigger`: `turn` or `speedup`.
- `turn_angle_deg`, `speed_ratio`, `out_speed`: how sharply the path turned, how
  much faster the ball got, and its outgoing speed in box heights per frame.
- `ball_to_box, ball_to_wrist`: distances divided by the hitter's box height.
- `wrist_speed`: a rough swing indicator, recorded but not used to accept hits.
- `ground_x, ground_y`: the hitter's ground point in the image.
- `court_x_m, court_y_m, side, behind_baseline_m, depth_zone, lateral_zone`: the
  hitter's **standing position**:
  - `side` is `far` or `near`.
  - `depth_zone` is `behind_baseline` (≥ 1 m behind), `baseline` (within 1 m),
    `mid_court`, or `net` (within 4 m of the net).
  - `lateral_zone` is the court width split into thirds, **as seen from the camera**.

`events.csv` = `hits.csv` plus `event_type, arm_raised, ball_above_head,
first_in_rally, contact_height_est_m`.
`serve_simulation/simulate_serves.py` reads this file.

## What "hit point" means here

| Meaning | Status |
|---|---|
| Time of the hit | **Estimated** (frame and timestamp) |
| Where the hitter stood | **Estimated** from the ground point + court homography |
| Contact height | **Rough estimate** from the pose scale (see above) |
| Exact 3D contact point / ball height on the racquet | **Not possible** with this setup. The court homography only covers the ground plane |

## Limitations

- These are heuristics. Bounces near a player, missed detections (the fast ball
  right after a serve is often blurred), camera cuts and lets can cause false or
  missed hits.
- Serve detection depends on the pose model seeing the raised arm and on the ball
  being detected near the contact.
- Hits are attributed to **track IDs**, which are not stable player identities.
- Tested only on synthetic data:
  - **A rally.** It had three hits, two bounces and one turn with no player nearby.
    All three hits were found and assigned correctly. The bounces and the far turn
    were rejected.
  - **A match.** It had a far-side serve (caught by `speedup`), a return, and a
    near-side serve. All three were found at the right frames and classified as
    serve / groundstroke / serve. The top of the toss was rejected.
  - **Not tested on real match data.**

## Possible upgrade: E2E-Spot

[E2E-Spot](https://github.com/jhong93/spot) (ECCV 2022) is a video event spotting
model with tennis checkpoints (for example `tennis_rny002gsm_gru_rgb`). It outputs
event frames, classes and scores. It could replace the rule-based hit timing and
serve labels. It is **not integrated**. It needs its own frame extraction and the
`config.json` that matches each checkpoint. It does not give a contact position and
does not know our track IDs.
