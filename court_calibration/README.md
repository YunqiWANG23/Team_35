# Court Calibration Module

This module finds the tennis court in our match videos. For every frame it predicts
**14 court keypoints**, such as the baseline corners, the singles sideline ends and
the service line ends. It can also fit a **homography** between those points and a
fixed reference court. Our app needs this to map positions on the court geometrically.

We don't train anything here. The model architecture, inference scripts and
post-processing come from
[yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector).
The scripts load that author's **pretrained weights** and only run inference. On top
of that, we added a script that renders a **top-down 2D court video**.

---

## Scope

**Implemented (in this folder's code)**

- Video inference (`infer_in_video.py`) and single-image inference (`infer_in_image.py`).
- Detection of 14 court keypoints from heatmaps (`tracknet.py`, `postprocess.postprocess`).
- Optional keypoint refinement. This crops around a predicted point, finds the court
  lines with classical CV and moves the point to where the lines cross
  (`postprocess.refine_kps`).
- Optional homography post-processing. This picks the best 4-point court configuration,
  computes a reference-court → image homography and projects all 14 reference keypoints
  back onto the frame. Points that are occluded or missing get filled in this way
  (`homography.py`, `court_reference.py`).
- Output: a copy of the input video with the keypoints drawn as red dots.
- **Top-down 2D court video** (`infer_top_down_video.py`, added by our team). Each
  frame is warped through the inverse homography (image → reference court) onto a
  bird's-eye view of the court, and the reference court lines are drawn on top.
  The output video shows only this top-down view.

**Not implemented yet (planned integration)**

- Detecting players and mapping their positions to top-down court coordinates. The
  top-down video warps the whole frame. It doesn't know where the players are.
- Player trajectories and court heatmaps.
- Exporting keypoints or homography matrices to a file. Right now they exist only
  inside the scripts and are not saved.

The homography in `homography.py` maps **reference court → image**.
`infer_top_down_video.py` uses its inverse. Player mapping could reuse the same
inverse once a player detector exists.

---

## File structure

```
court_calibration/
├── README.md
├── SOURCE.md             # attribution to the original repository
├── requirements.txt      # dependency pins inherited from the original project
├── infer_in_video.py     # video inference entry point
├── infer_in_image.py     # single-image inference entry point
├── infer_top_down_video.py  # top-down 2D court video (our addition)
├── tracknet.py           # network definition (BallTrackerNet, 15 output channels)
├── postprocess.py        # heatmap → point extraction, keypoint refinement
├── homography.py         # homography selection between detected and reference points
├── court_reference.py    # reference court geometry (keypoints, 4-point configurations)
├── utils.py              # line intersection and helper functions
├── weights/              # NOT in Git: create it and put court_model.pt here
│   └── court_model.pt
└── videos/               # NOT in Git: your own input videos (any location works)
```

`.gitignore` excludes `*.pt`, `*.pth`, `*.mp4`, `*.avi` and `*.mov`. Each teammate
has to get the weights and input videos themselves. The `weights/` folder is not
created for you.

---

## 1. Download the pretrained weights

Download the court detection weights from the original project:

https://drive.google.com/file/d/1f-Co64ehgq4uddcQm1aFBDtbnyZhQvgG/view

Save the file as:

```
court_calibration/weights/court_model.pt
```

Notes:

- This is the original project's **court keypoint network** (`BallTrackerNet` with
  `out_channels=15`), stored as a PyTorch `state_dict`. The scripts load it with
  `torch.load` + `load_state_dict`.
- It is **not** an Ultralytics model and **cannot** be loaded with `ultralytics.YOLO`.
- You only need these weights for inference. You do **not** need the original
  training dataset.

---

## 2. Install dependencies

Run all commands from the repository root:

```bash
cd court_calibration
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

`requirements.txt` is copied unchanged from the original project and has old pins
that may not install on a current setup:

- `torch==1.5.0+cu101` is a CUDA 10.1 build. It is not on PyPI, and you need
  PyTorch's own wheel index plus an old Python version (around 3.8) to install it.
  No build of it exists for macOS.
- `numpy==1.19.3` and `opencv_python==4.1.2.30` also need an old Python.
- `homography.py` uses `np.Inf`, which was removed in NumPy 2.0, so NumPy must be < 2.0.
- `tensorboardX` is only used for training and none of the inference files import it.

If the pinned install fails, install compatible versions of `torch`, `numpy<2`,
`opencv-python`, `scipy`, `sympy`, `tqdm` and `matplotlib` by hand. The inference
code imports exactly these. **No combination of versions has been tested in this
repository yet.**

---

## 3. Run inference

From the repository root:

```bash
cd court_calibration
mkdir -p output
```

**Basic video inference** (keypoints only):

```bash
python infer_in_video.py --model_path weights/court_model.pt --input_path videos/match.mp4 --output_path output/match_court.avi
```

**With homography post-processing**:

```bash
python infer_in_video.py --model_path weights/court_model.pt --input_path videos/match.mp4 --output_path output/match_court_homography.avi --use_homography
```

**With keypoint refinement and homography**:

```bash
python infer_in_video.py --model_path weights/court_model.pt --input_path videos/match.mp4 --output_path output/match_court_refined.avi --use_refine_kps --use_homography
```

`infer_in_video.py` takes these arguments:

| Argument | Required | Description |
|---|---|---|
| `--model_path` | yes (no default) | Path to the `.pt` weights |
| `--input_path` | yes (no default) | Input video |
| `--output_path` | yes (no default) | Output video |
| `--use_refine_kps` | flag | Refine keypoints with line intersections. Points 8, 9 and 12 are always skipped |
| `--use_homography` | flag | Replace the detected points with the reference keypoints projected through the best homography |

The argparse definitions do not mark any argument as required, but the script
crashes without them. `infer_in_image.py` takes the same arguments for a single image.

### Top-down 2D court video

```bash
python infer_top_down_video.py --model_path weights/court_model.pt --input_path videos/match.mp4 --output_path output/match_top_down.mp4
```

With keypoint refinement and temporal smoothing:

```bash
python infer_top_down_video.py --model_path weights/court_model.pt --input_path videos/match.mp4 --output_path output/match_top_down.mp4 --use_refine_kps --smooth 0.5
```

| Argument | Default | Description |
|---|---|---|
| `--model_path` | required | Path to the `.pt` weights |
| `--input_path` | required | Input video |
| `--output_path` | required | Output video. `.avi` uses the `DIVX` codec and anything else uses `mp4v`. The output folder is created if it's missing |
| `--use_refine_kps` | off | Same keypoint refinement as above |
| `--out_width` | `400` | Output width in pixels. The height follows the court's proportions (400 → 842) |
| `--max_error` | `20.0` | Rejects a homography if the keypoints not used to fit it reproject with a mean error above this (pixels, in 1280×720 space) |
| `--max_hold` | `15` | Number of frames to keep reusing the last good homography when the court isn't found (occlusion, replays) |
| `--smooth` | `0.0` | Exponential smoothing of the homography over time, in `[0, 1)`. Higher values reduce jitter but lag behind camera moves. `0` turns it off |
| `--no_court_lines` | off | Don't draw the reference court lines |
| `--device` | `auto` | `auto` (CUDA if available, else CPU), `cpu`, `cuda` or `mps` |

How it differs from `infer_in_video.py`:

- Frames are **streamed**, not loaded all at once, so memory use doesn't grow with
  video length.
- Every frame is resized to 1280×720 before detection. Inputs that aren't 1280×720
  therefore still get correctly placed points. Footage that isn't 16:9 is stretched
  before the model sees it, which may reduce accuracy.
- The homography is chosen the same way as in `homography.get_trans_matrix`. Ours
  also returns the reprojection error, checks all 14 points and avoids `np.Inf`,
  so it works with NumPy 2.
- Frames where no court is found and the hold period has ended are written as black
  frames with only the court lines. At the end the script prints how many frames
  were detected, reused (held) or had no court.

---

## 4. Check the results

The only output is the annotated video at `--output_path`, which has a red dot on
each detected keypoint. No coordinates, homography matrices or other data files are
written.

- The video is written with OpenCV's `DIVX` codec, so use an `.avi` output path.
  `.mp4` may not work, depending on your OpenCV build.
- OpenCV can fail **silently** if the output folder does not exist. That is why
  `mkdir -p output` comes before the commands above.
- The output FPS is the input FPS cut to an integer (29.97 becomes 29).
- With `--use_homography`, if no valid homography is found for a frame, that frame
  keeps its raw detections. Keypoints that were not detected are not drawn.

`infer_top_down_video.py` writes a portrait video with the court seen from above:
the net runs horizontally and the baselines are at the top and bottom. Keep these
in mind when reading it:

- The homography maps the **ground plane** only. Anything above the ground, such as
  players, the net, the umpire's chair or the stands, appears stretched and smeared
  away from the camera. A player's feet are the only part of them in the right place.
- Parts of the court the camera doesn't see stay black.
- Close-ups, replays and crowd shots usually have no court. They show as held or
  black frames.

---

## Runtime notes and limitations

The first three notes are about the original `infer_in_video.py` / `infer_in_image.py`.
`infer_top_down_video.py` handles device, resolution and memory differently, as
described above.

- **CPU / GPU:** the script uses CUDA if `torch.cuda.is_available()`, and the CPU
  otherwise. There is no flag to choose the device, and Apple MPS is not used. Frames
  are processed one at a time (batch size 1).
- **Resolution:** frames are resized to 640×360 for the network. The detected
  coordinates are then multiplied by a fixed factor of 2 (`postprocess(..., scale=2)`)
  and drawn on the original frame. The code therefore **assumes 1280×720 input**.
  Other resolutions run without errors, but the keypoints end up in the wrong place.
  Convert videos to 1280×720 first. Other aspect ratios are not handled.
- **Memory:** `infer_in_video.py` decodes the **whole video into RAM** before
  inference and keeps every annotated frame until it writes the output. One 1280×720 frame takes
  about 2.7 MB, so a 1-minute 30 fps clip needs about 5 GB. Use short clips or cut
  longer videos into pieces.
- **Speed:** we have not measured performance. Do not expect real-time speed,
  especially on CPU or with `--use_refine_kps`, which uses SymPy for line intersections.
- **Status:** the original scripts have not been run with the pretrained weights in
  this repository yet. `infer_top_down_video.py` was checked in two ways. First, on a
  synthetic court frame with known keypoints, the homography recovery and warp put
  97% of the court-line pixels within 2 px of the reference lines. Second, an
  end-to-end run on a synthetic 1920×1080 video with **randomly initialised** weights
  (CPU and MPS, `.mp4` and `.avi`) checked loading, streaming and writing. It has
  **not** been tested on real match footage with the pretrained weights. These
  checks used NumPy 2.0, OpenCV 4.13 and PyTorch 2.8, not the pins in
  `requirements.txt`.

---

## Acknowledgements

The model architecture, pretrained weights, inference scripts and post-processing
(keypoint refinement and homography) come from
[yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)
by yastrebksv. We did not train this model. We reuse the author's released weights
and the selected source files listed in `SOURCE.md`. `infer_top_down_video.py` is our
own addition, built on their model and homography code. See the original repository
for its licensing terms.
