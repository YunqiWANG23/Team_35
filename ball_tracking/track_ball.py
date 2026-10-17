"""Track the tennis ball in a video with the pretrained TrackNet model.

Wraps the original infer_on_video.py: frames are streamed instead of loaded into
memory, the device is selectable, and ball positions are written to a CSV in the
original video's pixel coordinates.

Output CSV columns: frame, time_s, x, y  (x, y empty when no ball was found)
"""
import argparse
import csv
import os
from collections import deque

import cv2
import numpy as np
import torch
from tqdm import tqdm

from model import BallTrackerNet
from general import postprocess
from infer_on_video import remove_outliers, split_track, interpolation

MODEL_WIDTH, MODEL_HEIGHT = 640, 360
# general.postprocess scales by 2, so raw detections are in 1280x720 space
TRACK_WIDTH, TRACK_HEIGHT = 1280, 720


def get_device(name):
    if name == 'auto':
        return 'cuda' if torch.cuda.is_available() else 'cpu'
    return name


def detect_ball(model, video_path, device):
    """ Run TrackNet on every frame (needs the 2 previous frames)
    :return
        ball_track: list of (x, y) in 1280x720 space, (None, None) if not found
        dists: distances between neighbouring detections, -1 if undefined
        fps, (width, height) of the original video
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError('Could not open input video: {}'.format(video_path))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    num_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) or None

    ball_track, dists = [], []
    window = deque(maxlen=3)
    with tqdm(total=num_frames) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            window.append(cv2.resize(frame, (MODEL_WIDTH, MODEL_HEIGHT)))
            pbar.update(1)
            if len(window) < 3:
                ball_track.append((None, None))
                dists.append(-1)
                continue

            # same input layout as infer_on_video.infer_model: current, previous, pre-previous
            imgs = np.concatenate((window[2], window[1], window[0]), axis=2)
            imgs = np.rollaxis(imgs.astype(np.float32) / 255.0, 2, 0)
            inp = torch.from_numpy(imgs).unsqueeze(0).to(device)
            with torch.no_grad():
                out = model(inp)
            output = out.argmax(dim=1).detach().cpu().numpy()
            x_pred, y_pred = postprocess(output)
            ball_track.append((x_pred, y_pred))

            if ball_track[-1][0] and ball_track[-2][0]:
                dists.append(float(np.hypot(ball_track[-1][0] - ball_track[-2][0],
                                            ball_track[-1][1] - ball_track[-2][1])))
            else:
                dists.append(-1)
    cap.release()
    return ball_track, dists, fps, size


def write_csv(path, ball_track, fps, scale_x, scale_y):
    out_dir = os.path.dirname(path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    with open(path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['frame', 'time_s', 'x', 'y'])
        for num, (x, y) in enumerate(ball_track):
            if x is None:
                writer.writerow([num, round(num / fps, 4), '', ''])
            else:
                writer.writerow([num, round(num / fps, 4), round(float(x) * scale_x, 2),
                                 round(float(y) * scale_y, 2)])


def write_video(video_path, path_out, ball_track, fps, scale_x, scale_y, trace=7):
    """ Second pass over the video: draw the last `trace` ball positions """
    out_dir = os.path.dirname(path_out)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    cap = cv2.VideoCapture(video_path)
    size = (int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)))
    ext = os.path.splitext(path_out)[1].lower()
    writer = cv2.VideoWriter(path_out, cv2.VideoWriter_fourcc(*('DIVX' if ext == '.avi' else 'mp4v')),
                             fps, size)
    if not writer.isOpened():
        raise RuntimeError('Could not open output video for writing: {}'.format(path_out))
    num = 0
    while True:
        ret, frame = cap.read()
        if not ret or num >= len(ball_track):
            break
        for i in range(trace):
            if num - i < 0 or ball_track[num - i][0] is None:
                break
            x = int(ball_track[num - i][0] * scale_x)
            y = int(ball_track[num - i][1] * scale_y)
            cv2.circle(frame, (x, y), radius=0, color=(0, 0, 255), thickness=10 - i)
        writer.write(frame)
        num += 1
    cap.release()
    writer.release()


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Track the tennis ball with TrackNet')
    parser.add_argument('--model_path', type=str, required=True, help='path to TrackNet weights')
    parser.add_argument('--input_path', type=str, required=True, help='path to input video')
    parser.add_argument('--output_csv', type=str, required=True, help='path to output CSV of ball positions')
    parser.add_argument('--output_video', type=str, default=None, help='optional annotated video (.mp4 or .avi)')
    parser.add_argument('--interpolate', action='store_true',
                        help='fill short gaps in the track (the --extrapolation option of infer_on_video.py)')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    args = parser.parse_args()

    device = get_device(args.device)
    model = BallTrackerNet()
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model = model.to(device)
    model.eval()

    ball_track, dists, fps, (width, height) = detect_ball(model, args.input_path, device)
    # remove_outliers looks at dists[i + 1]; a trailing -1 avoids an IndexError on the last frame
    ball_track = remove_outliers(ball_track, dists + [-1])
    if args.interpolate:
        for start, end in split_track(ball_track):
            detected = [i for i in range(start, end) if ball_track[i][0] is not None]
            if len(detected) < 2:
                continue
            # interpolation() also fills leading/trailing gaps with the edge value; only fill inner gaps
            first, last = detected[0], detected[-1] + 1
            ball_track[first:last] = interpolation(ball_track[first:last])

    scale_x, scale_y = width / TRACK_WIDTH, height / TRACK_HEIGHT
    write_csv(args.output_csv, ball_track, fps, scale_x, scale_y)
    if args.output_video:
        write_video(args.input_path, args.output_video, ball_track, fps, scale_x, scale_y)

    found = sum(1 for x, _ in ball_track if x is not None)
    print('Ball found in {} of {} frames'.format(found, len(ball_track)))
    print('Saved ball track to {}'.format(args.output_csv))
