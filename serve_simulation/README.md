# Serve Simulation Module (Part 2: Suggestion)

For every **serve event** found in a video, this module simulates what would change
if the player served with a different launch angle, aim, racquet speed, spin or
contact point. It then writes rule-based suggestions about which change makes the
serve more forgiving. The goal is to help players correct their **contact point**
and serve mechanics.

> **Everything here is a simulation under assumed parameters.** From one camera the
> video cannot measure racquet speed, spin or launch angle. Only the server's court
> position and a rough contact height come from the video. Results are labelled
> "Simulation" and should be shown that way in the app.

## Where the models come from

| Resource | What it is | How we use it |
|---|---|---|
| [orcn-dev/tennis-physics-3d-demo](https://github.com/orcn-dev/tennis-physics-3d-demo) | A JavaScript ball flight simulation with drag, Magnus lift, spin decay and RK4 integration. It has **no trained model or weights** | `physics.py` ports its flight model to Python/NumPy. See `SOURCE.md` and the header of `physics.py` for the changes |
| [TopSpin Ball Lab](https://topspin-labs.vercel.app/ball-lab) | A web page with sliders for racquet speed, launch angle, aim, spin (+top/−back) and contact height. We found **no public model, weights, API or source code** | We use the same parameter set. No code is used |

Neither resource is a "pretrained model" in the machine-learning sense. Both are
physics simulations or interactive tools.

## Files

```
serve_simulation/
├── README.md
├── SOURCE.md
├── requirements.txt
├── physics.py          # flight model (ported), vectorised over many shots
├── serve.py            # serve geometry, contact model, outcome, sweeps, windows, suggestions
├── simulate_serves.py  # batch: events.csv -> per-serve JSON + PNG
└── whatif.py           # before/after comparison for an interactive "adjust this serve" panel
```

## Parameters

We chose parameters a player can actually change in training:

| Parameter | Field | Baseline value | Swept |
|---|---|---|---|
| Racquet speed | `racquet_speed_ms` | 30 m/s (assumed, `--racquet_speed`) | ±2.5, ±5 m/s |
| Launch angle (+ up) | `launch_angle_deg` | solved per serve (see below) or `--launch_angle` | ±0.5, ±1, ±2° |
| Aim (+ = server's right) | `aim_deg` | 0 = towards the target box centre | ±2, ±4, ±6° |
| Spin (+ top / − back) | `spin_rpm` | 1500 rpm (assumed, `--spin_rpm`) | ±500 … ±1500 rpm |
| Contact height | `contact_height_m` | video estimate if 1.8–3.5 m, else 2.6 m | ±0.1, ±0.2, ±0.3 m |
| Contact in front of feet | `contact_forward_m` | 0.3 m | −0.3, +0.3, +0.6 m |

- **Racquet speed → ball speed.** Ball speed is computed as
  `v_ball = (1 + e_A) · v_racquet`, with apparent restitution `e_A = 0.4`. This
  simple impact model assumes the tossed ball is nearly at rest, so racquet speed
  and ball speed are not interchangeable. Change `e_A` with `--apparent_cor`.
- **Baseline launch angle.** It isn't measured, so by default it is solved: we pick
  the angle that lands the baseline serve about 1 m inside the service line. The
  question then becomes "if you served like this, how much room for error do you
  have, and what changes it?"
- **Target box.** The serve targets the box diagonally opposite the server. The
  side of the centre mark the server stands on picks deuce or ad.

## Outputs per serve (`serve_<frame>.json`)

- `baseline`:
  - `outcome`: `in`, `net`, `long`, `wide`, `centre`, or combinations such as
    `long_wide`.
  - `net_clearance_m`: how far above the net tape the ball passes.
  - Bounce point in court metres.
  - `box_margin_m`: distance to the nearest box line, positive when inside.
  - Ball speed at contact and at the bounce, flight time.
- `sweeps`: each parameter changed **one at a time** from the baseline, with the
  same metrics for every value.
- `launch_windows`: the **range of launch angles that land in the box**, for the
  baseline and for different contact heights, spins and racquet speeds. A wider
  window means a more forgiving serve. This is the main "how sensitive is it"
  measure, similar to Ball Lab's sensitivity idea.
- `suggestions`: rule-based text, each starting with "Simulation:". Examples:
  - a fix, when the baseline misses;
  - "contacting 0.3 m higher widens the window from 1.3° to 1.8°";
  - more topspin;
  - lower speed, including the speed trade-off;
  - how far the bounce moves per degree of launch angle;
  - the range of aim that stays in the box.
- `notes`: what came from the video and what was assumed.

`serve_<frame>.png` has two panels:
- **Side view:** the baseline serve and launch angle ±1° / ±2°, with the net and the
  service box depth.
- **Top view:** where the ball bounces for a grid of launch angle ±2° × aim ±6°,
  coloured by outcome. This shows how different serve angles change the result.

`serves_summary.json` has one line per serve: frame, time, track ID, side, contact
height, baseline outcome, window, and the top suggestion.

## Run

Run these from the repository root.

Batch, after `hit_detection/classify_events.py` has produced `events.csv`:

```bash
cd serve_simulation
pip install -r requirements.txt
```

```bash
python simulate_serves.py --events_csv ../output/events.csv --output_dir ../output/serves
```

Options:
- `--racquet_speed`, `--spin_rpm`, `--launch_angle`, `--contact_height` and
  `--contact_forward` override the assumptions.
- `--apparent_cor` sets the impact model.
- `--no_plots` skips the PNG files.

Interactive what-if (prints JSON with both serves and down-sampled trajectories for
the frontend):

```bash
python whatif.py --server_x 6.5 --server_y 24.0 --before '{"racquet_speed_ms": 30, "spin_rpm": 0, "contact_height_m": 2.4}' --after '{"contact_height_m": 2.7}'
```

`--after` keeps every other parameter, including the launch angle, the same. In this
example the same swing from 0.3 m higher lands **long**, while its launch-angle
window grows from 0.9° to 1.4°. So the player would need to hit slightly more
downward, but has more room for error.

## Physics model

- **Gravity, drag and Magnus lift:**
  - drag `F_d = −½ρA·C_D·|v|v` with `C_D = 0.55`;
  - Magnus `F_m = ½ρA·C_L·|v|(ω̂ × v)`;
  - lift coefficient `C_L = 1 / (2 + 1/S)`, where `S = Rω/|v|`. This is a common
    empirical fit. The demo's original `C_L = 0.2·min(1.2, S)` is available as
    `BallModel(lift_model='demo')`.
- **Spin:** decays as `ω0·e^(−t/4 s)`. Topspin makes the ball dip and backspin
  makes it float.
- **Integration:** RK4 with `dt = 2 ms` (4 ms for window scans). It stops at the
  first bounce, so there is no bounce or kick modelling.
- **Net:** 0.914 m at the centre, rising linearly to 1.07 m at the singles sticks.
  The ball radius is included in the clearance.

**Checks run:**
- Without drag, the range matches the analytic projectile (21.37 m simulated vs
  21.35 m analytic).
- More topspin shortens the bounce, and backspin lengthens it.
- A flat 151 km/h serve from 2.7 m needs about −5° to −4° launch angle, a
  plausible range.

**Not modelled:**
- sidespin (slice) and kick;
- what happens after the bounce;
- wind;
- player biomechanics;
- variation in ball and air properties.

The coefficients come from published fits and the demo. They have **not been
calibrated against measured serves**.

## Limitations

- Contact height from the video is a rough estimate (see `hit_detection/README.md`).
  Racquet speed, spin and launch angle are assumptions. The suggestions show how the
  outcome reacts to changes. They don't say what the player actually did.
- Suggestions are rules over the simulation, not a coaching model. They are not
  personalised beyond the serve's position and contact height.
- Tested only on synthetic serve events. Two serves (near side and far side) were
  simulated and produced JSON, PNG and suggestions. `run_pipeline.py` was also run
  end to end on a test video with random court and ball weights. **No real match
  footage has been tested.**
