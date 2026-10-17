"""Rule-based hit (stroke) detection from ball and player tracks.

A frame becomes a hit when:
  1. the ball's image trajectory turns sharply, with the vertical image direction
     reversing (towards / away from the camera), and
  2. the ball is close to a tracked person at that moment.

Each hit is assigned to the nearest person. If the player CSV contains court
coordinates, the hitter's court position (metres) and simple zones are reported.
These are heuristics to validate on real footage, not a trained hit detector.

Inputs:
  --ball_csv     from ball_tracking/track_ball.py
  --players_csv  from player_tracking/track_players.py
Outputs:
  --output_csv   one row per detected hit
  --summary_json per-track statistics
"""
import argparse
import json
import os

import numpy as np
import pandas as pd

COURT_WIDTH_M, COURT_LENGTH_M = 10.97, 23.77
NET_Y_M = COURT_LENGTH_M / 2
LEFT_WRIST, RIGHT_WRIST = 'left_wrist', 'right_wrist'


def load_ball(path):
    df = pd.read_csv(path)
    num = int(df['frame'].max()) + 1
    xy = np.full((num, 2), np.nan)
    xy[df['frame'].values, 0] = df['x'].values
    xy[df['frame'].values, 1] = df['y'].values
    last = df[df['time_s'] > 0].iloc[-1] if (df['time_s'] > 0).any() else None
    fps = float(last['frame'] / last['time_s']) if last is not None else 30.0
    return xy, fps


def fill_short_gaps(xy, max_gap):
    """ Linearly interpolate gaps of at most max_gap missing frames between two detections """
    xy = xy.copy()
    valid = ~np.isnan(xy[:, 0])
    idx = np.where(valid)[0]
    for a, b in zip(idx[:-1], idx[1:]):
        if 1 < b - a <= max_gap + 1:
            for t in range(a + 1, b):
                w = (t - a) / (b - a)
                xy[t] = (1 - w) * xy[a] + w * xy[b]
    return xy


def direction_changes(xy, k, min_move, min_angle, require_vertical_reversal):
    """ :return list of (frame, angle_deg) where the trajectory turns """
    candidates = []
    for t in range(k, len(xy) - k):
        before = xy[t] - xy[t - k]
        after = xy[t + k] - xy[t]
        if np.isnan(before).any() or np.isnan(after).any():
            continue
        nb, na = np.linalg.norm(before), np.linalg.norm(after)
        if nb < min_move or na < min_move:
            continue
        if require_vertical_reversal and before[1] * after[1] >= 0:
            continue
        angle = np.degrees(np.arccos(np.clip(np.dot(before, after) / (nb * na), -1.0, 1.0)))
        if angle >= min_angle:
            candidates.append((t, float(angle)))
    return candidates


def non_max_suppression(candidates, min_gap):
    kept = []
    for t, angle in sorted(candidates, key=lambda c: -c[1]):
        if all(abs(t - k) >= min_gap for k, _ in kept):
            kept.append((t, angle))
    return sorted(kept)


def box_distance(px, py, row):
    """ Distance from a point to a bounding box (0 if inside) """
    dx = max(row['x1'] - px, 0, px - row['x2'])
    dy = max(row['y1'] - py, 0, py - row['y2'])
    return float(np.hypot(dx, dy))


def wrist_distance(px, py, row):
    dists = []
    for w in (LEFT_WRIST, RIGHT_WRIST):
        if pd.notna(row.get(w + '_x')) and pd.notna(row.get(w + '_conf')) and row[w + '_conf'] >= 0.3:
            dists.append(np.hypot(row[w + '_x'] - px, row[w + '_y'] - py))
    return float(min(dists)) if dists else np.nan


def wrist_speed(players, track_id, frame, height, span=2):
    """ Max wrist displacement per frame around `frame`, divided by the person's box height """
    rows = players[(players['track_id'] == track_id) &
                   (players['frame'].between(frame - span, frame + span))].sort_values('frame')
    best = np.nan
    for w in (LEFT_WRIST, RIGHT_WRIST):
        if w + '_x' not in rows:
            continue
        pts = rows[[w + '_x', w + '_y']].values.astype(float)
        if len(pts) > 1:
            steps = np.linalg.norm(np.diff(pts, axis=0), axis=1)
            steps = steps[~np.isnan(steps)]
            if len(steps):
                best = np.nanmax([best, steps.max() / height])
    return best


def court_zones(court_x, court_y):
    """ Side, depth and lateral zones from a court position in metres (camera view) """
    side = 'far' if court_y < NET_Y_M else 'near'
    behind = -court_y if side == 'far' else court_y - COURT_LENGTH_M  # > 0 means behind own baseline
    to_net = abs(court_y - NET_Y_M)
    if behind >= 1.0:
        depth = 'behind_baseline'
    elif behind >= -1.0:
        depth = 'baseline'
    elif to_net <= 4.0:
        depth = 'net'
    else:
        depth = 'mid_court'
    third = COURT_WIDTH_M / 3
    lateral = 'left' if court_x < third else ('right' if court_x > 2 * third else 'center')
    return side, round(behind, 3), depth, lateral


def nearest_person(players_by_frame, frame, search):
    """ Rows of the frame closest to `frame` (within `search` frames) that has people """
    for d in range(search + 1):
        for f in (frame - d, frame + d) if d else (frame,):
            if f in players_by_frame:
                return players_by_frame[f]
    return None


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Rule-based tennis hit detection')
    parser.add_argument('--ball_csv', type=str, required=True)
    parser.add_argument('--players_csv', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--summary_json', type=str, default=None)
    parser.add_argument('--window', type=int, default=3, help='frames before/after used for ball direction')
    parser.add_argument('--max_gap', type=int, default=5, help='fill ball gaps up to this many frames')
    parser.add_argument('--min_move', type=float, default=2.0, help='min ball movement (px) over --window frames')
    parser.add_argument('--min_angle', type=float, default=45.0, help='min trajectory turn angle (degrees)')
    parser.add_argument('--any_turn', action='store_true',
                        help='accept turns without a vertical direction reversal (more bounce false positives)')
    parser.add_argument('--min_gap_s', type=float, default=0.4, help='min time between two hits (seconds)')
    parser.add_argument('--reach', type=float, default=0.6,
                        help='max ball-to-box distance as a fraction of the box height')
    parser.add_argument('--player_ids', type=str, default=None,
                        help='comma-separated track IDs to consider, e.g. "1,4" (manual player selection)')
    parser.add_argument('--include_all_people', action='store_true',
                        help='ignore the in_play_area column of the player CSV')
    args = parser.parse_args()

    ball_xy, fps = load_ball(args.ball_csv)
    ball_xy = fill_short_gaps(ball_xy, args.max_gap)

    players = pd.read_csv(args.players_csv)
    if args.player_ids:
        keep = [int(x) for x in args.player_ids.split(',')]
        players = players[players['track_id'].isin(keep)]
    if not args.include_all_people and players['in_play_area'].notna().any():
        players = players[players['in_play_area'] == 1]
    players_by_frame = {f: g for f, g in players.groupby('frame')}
    has_court = players['court_x_m'].notna().any()

    candidates = direction_changes(ball_xy, args.window, args.min_move, args.min_angle, not args.any_turn)
    candidates = non_max_suppression(candidates, max(1, int(round(args.min_gap_s * fps))))

    hits, rejected = [], 0
    for frame, angle in candidates:
        bx, by = ball_xy[frame]
        people = nearest_person(players_by_frame, frame, search=2)
        best = None
        if people is not None:
            for _, row in people.iterrows():
                height = max(row['y2'] - row['y1'], 1.0)
                dist = box_distance(bx, by, row) / height
                if dist <= args.reach and (best is None or dist < best[0]):
                    best = (dist, row, height)
        if best is None:
            rejected += 1
            continue

        dist, row, height = best
        hit = {
            'frame': frame,
            'time_s': round(frame / fps, 3),
            'track_id': int(row['track_id']),
            'ball_x': round(float(bx), 2),
            'ball_y': round(float(by), 2),
            'turn_angle_deg': round(angle, 1),
            'ball_to_box': round(dist, 3),
            'ball_to_wrist': round(wrist_distance(bx, by, row) / height, 3),
            'wrist_speed': round(float(wrist_speed(players, row['track_id'], frame, height)), 4),
            'ground_x': row['ground_x'],
            'ground_y': row['ground_y'],
            'court_x_m': '', 'court_y_m': '', 'side': '', 'behind_baseline_m': '',
            'depth_zone': '', 'lateral_zone': '',
        }
        if has_court and pd.notna(row['court_x_m']):
            side, behind, depth, lateral = court_zones(row['court_x_m'], row['court_y_m'])
            hit.update({'court_x_m': row['court_x_m'], 'court_y_m': row['court_y_m'], 'side': side,
                        'behind_baseline_m': behind, 'depth_zone': depth, 'lateral_zone': lateral})
        hits.append(hit)

    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    columns = ['frame', 'time_s', 'track_id', 'ball_x', 'ball_y', 'turn_angle_deg', 'ball_to_box',
               'ball_to_wrist', 'wrist_speed', 'ground_x', 'ground_y', 'court_x_m', 'court_y_m', 'side',
               'behind_baseline_m', 'depth_zone', 'lateral_zone']
    hits_df = pd.DataFrame(hits, columns=columns)
    hits_df.to_csv(args.output_csv, index=False)

    if args.summary_json:
        summary = {'fps': round(fps, 3), 'trajectory_turns': len(candidates), 'hits': len(hits),
                   'turns_without_player_nearby': rejected, 'players': {}}
        for track_id, g in hits_df.groupby('track_id'):
            entry = {'hits': int(len(g)), 'hit_times_s': g['time_s'].tolist()}
            located = g[g['depth_zone'] != '']
            if len(located):
                entry['side'] = located['side'].value_counts().to_dict()
                entry['depth_zone'] = located['depth_zone'].value_counts().to_dict()
                entry['lateral_zone'] = located['lateral_zone'].value_counts().to_dict()
                entry['mean_behind_baseline_m'] = round(float(located['behind_baseline_m'].astype(float).mean()), 3)
            summary['players'][str(track_id)] = entry
        sdir = os.path.dirname(args.summary_json)
        if sdir:
            os.makedirs(sdir, exist_ok=True)
        with open(args.summary_json, 'w') as f:
            json.dump(summary, f, indent=2)

    print('Trajectory turns: {}, hits: {}, turns without a player nearby: {}'.format(
        len(candidates), len(hits), rejected))
    print('Saved hits to {}'.format(args.output_csv))
