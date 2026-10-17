"""Render a top-down (bird's-eye) 2D view of the tennis court from a video.

For every frame the court keypoints are detected, a homography between the
reference court and the frame is estimated, and the frame is warped onto the
reference court plane. The output video only contains the top-down view.
"""
import argparse
import os

import cv2
import numpy as np
import torch
from scipy.spatial import distance
from tqdm import tqdm

from tracknet import BallTrackerNet
from postprocess import postprocess, refine_kps
from homography import court_ref, court_conf_ind, refer_kps

MODEL_WIDTH, MODEL_HEIGHT = 640, 360
# postprocess() scales heatmap coordinates by 2, so keypoints live in 1280x720 space
FRAME_WIDTH, FRAME_HEIGHT = 1280, 720
NUM_KPS = 14


def detect_keypoints(model, frame, device, use_refine_kps=False):
    """ Detect 14 court keypoints in a 1280x720 BGR frame
    :return
        points: list of 14 (x, y) tuples, (None, None) for undetected points
    """
    img = cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT))
    inp = np.rollaxis(img.astype(np.float32) / 255., 2, 0)
    inp = torch.from_numpy(inp).unsqueeze(0).to(device)
    with torch.no_grad():
        out = model(inp)[0]
    pred = torch.sigmoid(out).cpu().numpy()

    points = []
    for kps_num in range(NUM_KPS):
        heatmap = (pred[kps_num] * 255).astype(np.uint8)
        x_pred, y_pred = postprocess(heatmap, low_thresh=170, max_radius=25)
        if use_refine_kps and kps_num not in [8, 12, 9] and x_pred and y_pred:
            x_pred, y_pred = refine_kps(frame, int(y_pred), int(x_pred))
        points.append((x_pred, y_pred))
    return points


def find_homography(points):
    """ Same selection as homography.get_trans_matrix, but also returns the error
    :return
        matrix: 3x3 reference court -> frame homography, or None
        error: mean reprojection error (pixels) of the keypoints not used to fit it
    """
    best_matrix, best_error = None, np.inf
    for conf_ind, inds in court_conf_ind.items():
        inters = [points[i] for i in inds]
        if any(None in p for p in inters):
            continue
        matrix, _ = cv2.findHomography(np.float32(court_ref.court_conf[conf_ind]),
                                       np.float32(inters), method=0)
        if matrix is None:
            continue
        trans_kps = cv2.perspectiveTransform(refer_kps, matrix).reshape(-1, 2)
        dists = [distance.euclidean(points[i], trans_kps[i]) for i in range(NUM_KPS)
                 if i not in inds and points[i][0] is not None]
        if not dists:
            continue
        error = np.mean(dists)
        if error < best_error:
            best_matrix, best_error = matrix, error
    return best_matrix, best_error


def draw_court_lines(canvas, scale, color=(255, 255, 255), thickness=1):
    lines = court_ref.get_important_lines()
    for i in range(0, len(lines), 2):
        p1 = (int(round(lines[i][0] * scale)), int(round(lines[i][1] * scale)))
        p2 = (int(round(lines[i + 1][0] * scale)), int(round(lines[i + 1][1] * scale)))
        cv2.line(canvas, p1, p2, color, thickness)
    return canvas


def warp_to_top_down(frame, matrix, scale, out_size):
    """ Warp a 1280x720 frame onto the (scaled) reference court plane
    :params
        matrix: reference court -> frame homography
        scale: reference court pixels -> output pixels
        out_size: (width, height) of the output image
    """
    scale_mat = np.diag([scale, scale, 1.0])
    frame_to_top = scale_mat @ np.linalg.inv(matrix)
    return cv2.warpPerspective(frame, frame_to_top, out_size)


def open_writer(path, fps, size):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    ext = os.path.splitext(path)[1].lower()
    fourcc = cv2.VideoWriter_fourcc(*('DIVX' if ext == '.avi' else 'mp4v'))
    writer = cv2.VideoWriter(path, fourcc, fps, size)
    if not writer.isOpened():
        raise RuntimeError('Could not open output video for writing: {}'.format(path))
    return writer


def get_device(name):
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return name


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Render a top-down 2D court view video')
    parser.add_argument('--model_path', type=str, required=True, help='path to model')
    parser.add_argument('--input_path', type=str, required=True, help='path to input video')
    parser.add_argument('--output_path', type=str, required=True, help='path to output video (.mp4 or .avi)')
    parser.add_argument('--use_refine_kps', action='store_true', help='whether to use refine kps postprocessing')
    parser.add_argument('--out_width', type=int, default=400, help='width of the top-down video in pixels')
    parser.add_argument('--max_error', type=float, default=20.0,
                        help='reject homographies with a mean reprojection error above this (pixels, 1280x720)')
    parser.add_argument('--max_hold', type=int, default=15,
                        help='reuse the last good homography for up to this many frames when detection fails')
    parser.add_argument('--smooth', type=float, default=0.0,
                        help='temporal smoothing of the homography in [0, 1); 0 disables it')
    parser.add_argument('--no_court_lines', action='store_true', help='do not draw reference court lines')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    args = parser.parse_args()

    if not 0.0 <= args.smooth < 1.0:
        parser.error('--smooth must be in [0, 1)')

    device = get_device(args.device)
    model = BallTrackerNet(out_channels=15)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    cap = cv2.VideoCapture(args.input_path)
    if not cap.isOpened():
        raise RuntimeError('Could not open input video: {}'.format(args.input_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    scale = args.out_width / court_ref.court_total_width
    out_w = args.out_width // 2 * 2
    out_h = int(round(court_ref.court_total_height * scale)) // 2 * 2
    writer = open_writer(args.output_path, fps, (out_w, out_h))

    stats = {'detected': 0, 'held': 0, 'missing': 0}
    last_matrix, smoothed, held = None, None, 0
    with tqdm(total=num_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame = cv2.resize(frame, (FRAME_WIDTH, FRAME_HEIGHT))
            points = detect_keypoints(model, frame, device, args.use_refine_kps)
            matrix, error = find_homography(points)

            if matrix is not None and error <= args.max_error:
                matrix = matrix / matrix[2, 2]
                if smoothed is None or args.smooth == 0.0:
                    smoothed = matrix
                else:
                    smoothed = args.smooth * smoothed + (1 - args.smooth) * matrix
                last_matrix, held = smoothed, 0
                stats['detected'] += 1
            elif last_matrix is not None and held < args.max_hold:
                held += 1
                stats['held'] += 1
            else:
                last_matrix, smoothed = None, None
                stats['missing'] += 1

            if last_matrix is not None:
                top_down = warp_to_top_down(frame, last_matrix, scale, (out_w, out_h))
            else:
                top_down = np.zeros((out_h, out_w, 3), dtype=np.uint8)
            if not args.no_court_lines:
                draw_court_lines(top_down, scale)
            writer.write(top_down)
            pbar.update(1)

    cap.release()
    writer.release()
    print('Frames with court detected: {detected}, reused last homography: {held}, '
          'no court: {missing}'.format(**stats))
    print('Saved top-down video to {}'.format(args.output_path))
