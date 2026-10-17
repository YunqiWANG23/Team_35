"""Classify detected hits into events (serve / overhead / groundstroke) and
estimate the contact height.

Rules, applied to the hitter's pose at the hit frame:
  - arm_raised:      a confident wrist is above the nose
  - ball_above_head: the ball is above the nose in the image
  - first_in_rally:  no other hit in the previous --rally_gap_s seconds
  - serve        = arm_raised and ball_above_head and first_in_rally
                   (and, if court positions exist, the hitter is not inside the court
                   by more than --max_inside_m)
  - overhead     = arm_raised and ball_above_head, but not a serve
  - groundstroke = everything else (includes volleys)

Contact height (metres) is a rough single-camera estimate. The vertical pixel
distance from the hitter's ground point to the ball is converted to metres using the
ground-to-shoulder distance as a scale, assuming the shoulders are at
--shoulder_ratio x --player_height. It ignores perspective and the ball's depth
offset from the player, so treat it as approximate.

Inputs:  hits.csv (detect_hits.py), players.csv (player_tracking)
Output:  events.csv = hits.csv + event columns
"""
import argparse
import os

import numpy as np
import pandas as pd

KP_CONF = 0.3


def kp(row, name):
    """ (x, y) of a keypoint if it is confident, else None """
    x, y, c = row.get(name + '_x'), row.get(name + '_y'), row.get(name + '_conf')
    if pd.isna(x) or pd.isna(y) or pd.isna(c) or c < KP_CONF:
        return None
    return float(x), float(y)


def contact_height(row, ball_y, player_height, shoulder_ratio):
    shoulders = [p for p in (kp(row, 'left_shoulder'), kp(row, 'right_shoulder')) if p]
    if not shoulders or pd.isna(row['ground_y']):
        return np.nan
    shoulder_y = np.mean([p[1] for p in shoulders])
    pixels = row['ground_y'] - shoulder_y
    if pixels <= 1:
        return np.nan
    metres_per_pixel = shoulder_ratio * player_height / pixels
    return float((row['ground_y'] - ball_y) * metres_per_pixel)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Classify hits into serve / overhead / groundstroke')
    parser.add_argument('--hits_csv', type=str, required=True)
    parser.add_argument('--players_csv', type=str, required=True)
    parser.add_argument('--output_csv', type=str, required=True)
    parser.add_argument('--rally_gap_s', type=float, default=3.0,
                        help='a hit with no other hit this long before it starts a rally')
    parser.add_argument('--max_inside_m', type=float, default=1.0,
                        help='a serve must be hit from no more than this far inside the baseline')
    parser.add_argument('--player_height', type=float, default=1.80, help='assumed player height (m)')
    parser.add_argument('--shoulder_ratio', type=float, default=0.82,
                        help='shoulder height as a fraction of body height')
    args = parser.parse_args()

    hits = pd.read_csv(args.hits_csv).sort_values('frame').reset_index(drop=True)
    players = pd.read_csv(args.players_csv)
    rows = {(int(r.frame), int(r.track_id)): r for r in players.itertuples(index=False)}
    columns = list(players.columns)

    events = []
    prev_time = None
    for hit in hits.itertuples(index=False):
        match = rows.get((int(hit.frame), int(hit.track_id)))
        row = dict(zip(columns, match)) if match is not None else {}

        nose = kp(row, 'nose') if row else None
        wrists = [p for p in (kp(row, 'left_wrist'), kp(row, 'right_wrist')) if p] if row else []
        arm_raised = bool(nose and wrists and min(w[1] for w in wrists) < nose[1])
        ball_above_head = bool(nose and hit.ball_y < nose[1])
        first_in_rally = prev_time is None or hit.time_s - prev_time >= args.rally_gap_s
        prev_time = hit.time_s

        behind = getattr(hit, 'behind_baseline_m', np.nan)
        position_ok = pd.isna(behind) or behind >= -args.max_inside_m

        if arm_raised and ball_above_head and first_in_rally and position_ok:
            event = 'serve'
        elif arm_raised and ball_above_head:
            event = 'overhead'
        else:
            event = 'groundstroke'

        height = contact_height(row, hit.ball_y, args.player_height, args.shoulder_ratio) if row else np.nan
        events.append({'event_type': event, 'arm_raised': int(arm_raised),
                       'ball_above_head': int(ball_above_head), 'first_in_rally': int(first_in_rally),
                       'contact_height_est_m': round(height, 2) if not np.isnan(height) else np.nan})

    event_columns = ['event_type', 'arm_raised', 'ball_above_head', 'first_in_rally', 'contact_height_est_m']
    out = pd.concat([hits, pd.DataFrame(events, index=hits.index, columns=event_columns)], axis=1)
    out_dir = os.path.dirname(args.output_csv)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    out.to_csv(args.output_csv, index=False)

    counts = out['event_type'].value_counts().to_dict() if len(out) else {}
    print('Events: {}'.format(counts))
    print('Saved events to {}'.format(args.output_csv))
