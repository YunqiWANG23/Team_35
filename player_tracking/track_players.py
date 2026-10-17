"""Detect and track people in a tennis video with YOLO11 Pose + ByteTrack.

Writes one CSV row per tracked person per frame. If a court homography file from
court_calibration/export_homography.py is given, each person's ground point is
also mapped to court coordinates in metres, and people outside the playing area
(umpire, ball kids, crowd) can be flagged or dropped.

A track ID is NOT a long-term player identity: it can change after occlusions,
camera cuts or change of ends.
"""
import argparse
import csv
import json
import os

import cv2
import numpy as np
from ultralytics import YOLO

# COCO keypoint order used by YOLO pose models
KEYPOINT_NAMES = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                  'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                  'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                  'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
LEFT_ANKLE, RIGHT_ANKLE = 15, 16
COURT_WIDTH_M, COURT_LENGTH_M = 10.97, 23.77


def load_homographies(path):
    """ :return list indexed by frame of 3x3 image -> court (metres) matrices or None """
    with open(path) as f:
        data = json.load(f)
    return [None if fr['image_to_court'] is None else np.array(fr['image_to_court'])
            for fr in data['frames']]


def ground_point(box, kps_xy, kps_conf, min_kp_conf=0.5):
    """ Approximate where the person touches the ground (image pixels)
    Uses the midpoint of the ankles if both are confident, else the bottom centre of the box.
    """
    if kps_conf is not None and kps_conf[LEFT_ANKLE] >= min_kp_conf and kps_conf[RIGHT_ANKLE] >= min_kp_conf:
        x = (kps_xy[LEFT_ANKLE][0] + kps_xy[RIGHT_ANKLE][0]) / 2
        y = (kps_xy[LEFT_ANKLE][1] + kps_xy[RIGHT_ANKLE][1]) / 2
        return float(x), float(y), 'ankles'
    return float((box[0] + box[2]) / 2), float(box[3]), 'box_bottom'


def to_court(matrix, x, y):
    p = matrix @ np.array([x, y, 1.0])
    return float(p[0] / p[2]), float(p[1] / p[2])


def in_play_area(cx, cy, margin_side, margin_end):
    return (-margin_side <= cx <= COURT_WIDTH_M + margin_side and
            -margin_end <= cy <= COURT_LENGTH_M + margin_end)


def csv_header():
    header = ['frame', 'track_id', 'conf', 'x1', 'y1', 'x2', 'y2',
              'ground_x', 'ground_y', 'ground_source', 'court_x_m', 'court_y_m', 'in_play_area']
    for name in KEYPOINT_NAMES:
        header += [name + '_x', name + '_y', name + '_conf']
    return header


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Track people with YOLO11 Pose + ByteTrack')
    parser.add_argument('--input_path', type=str, required=True, help='path to input video')
    parser.add_argument('--output_csv', type=str, required=True, help='path to output CSV')
    parser.add_argument('--model', type=str, default='yolo11s-pose.pt',
                        help='Ultralytics pose model name or path (downloaded automatically on first use)')
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml', help='Ultralytics tracker config')
    parser.add_argument('--conf', type=float, default=0.25, help='detection confidence threshold')
    parser.add_argument('--imgsz', type=int, default=640, help='inference image size')
    parser.add_argument('--court_homography', type=str, default=None,
                        help='JSON from court_calibration/export_homography.py')
    parser.add_argument('--margin_side', type=float, default=3.0,
                        help='metres beyond the doubles sidelines still counted as play area')
    parser.add_argument('--margin_end', type=float, default=6.0,
                        help='metres beyond the baselines still counted as play area')
    parser.add_argument('--players_only', action='store_true',
                        help='drop people outside the play area (needs --court_homography)')
    parser.add_argument('--output_video', type=str, default=None, help='optional annotated video')
    parser.add_argument('--device', type=str, default=None, help='e.g. cpu, 0 (CUDA), mps; default: auto')
    args = parser.parse_args()

    if args.players_only and not args.court_homography:
        parser.error('--players_only needs --court_homography')
    homographies = load_homographies(args.court_homography) if args.court_homography else None

    model = YOLO(args.model)
    results = model.track(source=args.input_path, tracker=args.tracker, classes=[0], conf=args.conf,
                          imgsz=args.imgsz, device=args.device, stream=True, persist=True,
                          save=False, verbose=False)

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    video_writer = None
    num_frames, num_rows, track_ids = 0, 0, set()

    with open(args.output_csv, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(csv_header())
        for frame_num, result in enumerate(results):
            num_frames += 1
            if args.output_video:
                annotated = result.plot()
                if video_writer is None:
                    vdir = os.path.dirname(args.output_video)
                    if vdir:
                        os.makedirs(vdir, exist_ok=True)
                    ext = os.path.splitext(args.output_video)[1].lower()
                    fps = cv2.VideoCapture(args.input_path).get(cv2.CAP_PROP_FPS) or 30.0
                    video_writer = cv2.VideoWriter(
                        args.output_video, cv2.VideoWriter_fourcc(*('DIVX' if ext == '.avi' else 'mp4v')),
                        fps, (annotated.shape[1], annotated.shape[0]))
                video_writer.write(annotated)

            boxes = result.boxes
            if boxes is None or boxes.id is None:
                continue
            ids = boxes.id.int().cpu().tolist()
            xyxy = boxes.xyxy.cpu().numpy()
            confs = boxes.conf.cpu().numpy()
            if result.keypoints is not None:
                kps_xy = result.keypoints.xy.cpu().numpy()
                kps_conf = result.keypoints.conf.cpu().numpy() if result.keypoints.conf is not None else None
            else:
                kps_xy, kps_conf = None, None

            matrix = None
            if homographies is not None and frame_num < len(homographies):
                matrix = homographies[frame_num]

            for i, track_id in enumerate(ids):
                kxy = kps_xy[i] if kps_xy is not None else None
                kconf = kps_conf[i] if kps_conf is not None else None
                gx, gy, source = ground_point(xyxy[i], kxy, kconf)

                court_x, court_y, in_area = '', '', ''
                if matrix is not None:
                    court_x, court_y = to_court(matrix, gx, gy)
                    in_area = int(in_play_area(court_x, court_y, args.margin_side, args.margin_end))
                    court_x, court_y = round(court_x, 3), round(court_y, 3)
                if args.players_only and in_area != 1:
                    continue

                row = [frame_num, track_id, round(float(confs[i]), 4),
                       *[round(float(v), 2) for v in xyxy[i]],
                       round(gx, 2), round(gy, 2), source, court_x, court_y, in_area]
                for k in range(len(KEYPOINT_NAMES)):
                    if kxy is None:
                        row += ['', '', '']
                    else:
                        row += [round(float(kxy[k][0]), 2), round(float(kxy[k][1]), 2),
                                '' if kconf is None else round(float(kconf[k]), 4)]
                writer.writerow(row)
                num_rows += 1
                track_ids.add(track_id)

    if video_writer is not None:
        video_writer.release()
    print('Processed {} frames, wrote {} rows for {} track IDs'.format(num_frames, num_rows, len(track_ids)))
    print('Saved player tracks to {}'.format(args.output_csv))
