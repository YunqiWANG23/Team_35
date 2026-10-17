# Court Calibration Module

This module finds the tennis court in our match videos. For every frame it predicts
**14 court keypoints**, such as the baseline corners, the singles sideline ends and
the service line ends. It can also fit a **homography** between those points and a
fixed reference court. Our app needs this to map positions on the court geometrically.

We don't train anything here. The model architecture, inference scripts and
post-processing come from
[yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector).
The scripts load that author's **pretrained weights** and only run inference.

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

**Not implemented yet (planned integration)**

- Mapping player positions into top-down (bird's-eye) court coordinates.
- Player trajectories and court heatmaps.
- Exporting keypoints or homography matrices to a file. Right now they exist only
  inside the script and are not saved.

The homography in `homography.py` maps **reference court → image**. Mapping players
onto a top-down court would need its inverse, plus a player detector. Neither one
exists in this folder yet.

---

## File structure

```
court_calibration/
├── README.md
├── SOURCE.md             # attribution to the original repository
├── requirements.txt      # dependency pins inherited from the original project
├── infer_in_video.py     # video inference entry point
├── infer_in_image.py     # single-image inference entry point
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

---

## Runtime notes and limitations

- **CPU / GPU:** the script uses CUDA if `torch.cuda.is_available()`, and the CPU
  otherwise. There is no flag to choose the device, and Apple MPS is not used. Frames
  are processed one at a time (batch size 1).
- **Resolution:** frames are resized to 640×360 for the network. The detected
  coordinates are then multiplied by a fixed factor of 2 (`postprocess(..., scale=2)`)
  and drawn on the original frame. The code therefore **assumes 1280×720 input**.
  Other resolutions run without errors, but the keypoints end up in the wrong place.
  Convert videos to 1280×720 first. Other aspect ratios are not handled.
- **Memory:** the script decodes the **whole video into RAM** before inference and
  keeps every annotated frame until it writes the output. One 1280×720 frame takes
  about 2.7 MB, so a 1-minute 30 fps clip needs about 5 GB. Use short clips or cut
  longer videos into pieces.
- **Speed:** we have not measured performance. Do not expect real-time speed,
  especially on CPU or with `--use_refine_kps`, which uses SymPy for line intersections.
- **Status:** this README was written from reading the code. The scripts have not
  been run with the weights in this repository yet.

---

## Acknowledgements

The model architecture, pretrained weights, inference scripts and post-processing
(keypoint refinement and homography) come from
[yastrebksv/TennisCourtDetector](https://github.com/yastrebksv/TennisCourtDetector)
by yastrebksv. We did not train this model. We reuse the author's released weights
and the selected source files listed in `SOURCE.md`. See the original repository for
its licensing terms.
