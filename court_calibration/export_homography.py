"""Export a per-frame image -> court homography for a video.

The matrices map pixel coordinates of the ORIGINAL video directly to court
coordinates in metres. These are ground-plane coordinates: x runs across the
court from the far-left doubles corner (0 to 10.97 m) and y runs from the far
baseline (0 m) to the near baseline (23.77 m). The net is at y = 11.885 m.
"Far" and "left" are as seen from the camera.

Other modules (player_tracking, hit_detection) read this JSON file.
"""
import argparse
import json
import os

import cv2
import numpy as np
import torch
from tqdm import tqdm

from tracknet import BallTrackerNet
from court_reference import CourtReference
from infer_top_down_video import (detect_keypoints, find_homography, get_device,
                                  FRAME_WIDTH, FRAME_HEIGHT)

COURT_WIDTH_M = 10.97   # doubles width
COURT_LENGTH_M = 23.77  # baseline to baseline


def reference_to_metres():
    """ 3x3 matrix: reference court pixels -> metres (origin at far-left doubles corner) """
    ref = CourtReference()
    (x0, y0), (x1, _) = ref.baseline_top
    y1 = ref.baseline_bottom[0][1]
    sx = COURT_WIDTH_M / (x1 - x0)
    sy = COURT_LENGTH_M / (y1 - y0)
    return np.array([[sx, 0, -x0 * sx], [0, sy, -y0 * sy], [0, 0, 1]], dtype=np.float64)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Export per-frame image -> court (metres) homographies')
    parser.add_argument('--model_path', type=str, required=True, help='path to court model')
    parser.add_argument('--input_path', type=str, required=True, help='path to input video')
    parser.add_argument('--output_json', type=str, required=True, help='path to output JSON')
    parser.add_argument('--use_refine_kps', action='store_true', help='whether to use refine kps postprocessing')
    parser.add_argument('--max_error', type=float, default=20.0,
                        help='reject homographies with a mean reprojection error above this (pixels, 1280x720)')
    parser.add_argument('--max_hold', type=int, default=15,
                        help='reuse the last good homography for up to this many frames when detection fails')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    args = parser.parse_args()

    device = get_device(args.device)
    model = BallTrackerNet(out_channels=15)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(args.input_path)
    if not cap.isOpened():
        raise RuntimeError('Could not open input video: {}'.format(args.input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    # original pixels -> 1280x720 pixels (the space keypoints are detected in)
    to_work = np.diag([FRAME_WIDTH / width, FRAME_HEIGHT / height, 1.0])
    to_metres = reference_to_metres()

    frames = []
    last_matrix, held = None, 0
    with tqdm(total=num_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            points = detect_keypoints(model, frame, device, args.use_refine_kps)
            matrix, error = find_homography(points)

            if matrix is not None and error <= args.max_error:
                last_matrix = to_metres @ np.linalg.inv(matrix) @ to_work
                last_matrix = last_matrix / last_matrix[2, 2]
                held, status = 0, 'detected'
            elif last_matrix is not None and held < args.max_hold:
                held, status = held + 1, 'held'
            else:
                last_matrix, status, error = None, 'missing', None

            frames.append({
                'frame': len(frames),
                'status': status,
                'reprojection_error': None if status != 'detected' else round(float(error), 3),
                'image_to_court': None if last_matrix is None else np.round(last_matrix, 10).tolist(),
            })
            pbar.update(1)
    cap.release()

    out_dir = os.path.dirname(args.output_json)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(args.output_json, 'w') as f:
        json.dump({
            'video': os.path.basename(args.input_path),
            'fps': fps,
            'width': width,
            'height': height,
            'court_units': 'metres',
            'court_size_m': [COURT_WIDTH_M, COURT_LENGTH_M],
            'frames': frames,
        }, f)

    counts = {s: sum(1 for x in frames if x['status'] == s) for s in ('detected', 'held', 'missing')}
    print('Frames with court detected: {detected}, reused last homography: {held}, '
          'no court: {missing}'.format(**counts))
    print('Saved homographies to {}'.format(args.output_json))
