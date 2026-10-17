"""Run serve what-if simulations for every serve event of a video.

Input:  events.csv from hit_detection/classify_events.py
Output (in --output_dir):
  serve_<frame>.json  baseline, one-at-a-time sweeps, launch-angle windows, suggestions
  serve_<frame>.png   side view (launch-angle changes) + top view (launch angle x aim grid)
  serves_summary.json one entry per serve

What comes from the video: server court position (if the court was calibrated) and an
estimated contact height. Racquet speed, spin and launch angle are NOT measured; they
are assumptions (CLI defaults or overrides), and the baseline launch angle is solved so
the baseline serve lands about 1 m inside the service line unless --launch_angle is given.
"""
import argparse
import json
import os
from dataclasses import replace

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from serve import (Server, ServeParams, analyse_serve, evaluate, NET_Y, CENTER_X, SINGLES_LEFT,
                   SINGLES_RIGHT, SERVICE_DEPTH, COURT_WIDTH, COURT_LENGTH, net_height)

OUTCOME_COLORS = {'in': '#2e7d32', 'net': '#c62828', 'long': '#ef6c00', 'wide': '#6a1b9a',
                  'centre': '#1565c0'}


def outcome_color(outcome):
    return OUTCOME_COLORS.get(outcome.split('_')[0], '#616161')


def draw_court(ax):
    lines = [((0, 0), (COURT_WIDTH, 0)), ((0, COURT_LENGTH), (COURT_WIDTH, COURT_LENGTH)),
             ((0, 0), (0, COURT_LENGTH)), ((COURT_WIDTH, 0), (COURT_WIDTH, COURT_LENGTH)),
             ((SINGLES_LEFT, 0), (SINGLES_LEFT, COURT_LENGTH)), ((SINGLES_RIGHT, 0), (SINGLES_RIGHT, COURT_LENGTH)),
             ((SINGLES_LEFT, NET_Y - SERVICE_DEPTH), (SINGLES_RIGHT, NET_Y - SERVICE_DEPTH)),
             ((SINGLES_LEFT, NET_Y + SERVICE_DEPTH), (SINGLES_RIGHT, NET_Y + SERVICE_DEPTH)),
             ((CENTER_X, NET_Y - SERVICE_DEPTH), (CENTER_X, NET_Y + SERVICE_DEPTH))]
    for (x0, y0), (x1, y1) in lines:
        ax.plot([x0, x1], [y0, y1], color='#9e9e9e', lw=1)
    ax.plot([-0.9, COURT_WIDTH + 0.9], [NET_Y, NET_Y], color='black', lw=2)


def plot_serve(path, server, analysis, model=None):
    base = ServeParams(**analysis['baseline']['params'])
    fig, (ax_side, ax_top) = plt.subplots(1, 2, figsize=(13, 6), gridspec_kw={'width_ratios': [1.6, 1]})

    # side view: launch angle changes, distance along the serve direction vs height
    deltas = [-2, -1, 0, 1, 2]
    variants = [replace(base, launch_angle_deg=base.launch_angle_deg + d) for d in deltas]
    net_dist = abs(server.y - NET_Y)
    for d, r in zip(deltas, evaluate(server, variants, model=model, keep_traj=True)):
        tr = r['trajectory']
        dist = np.abs(tr[:, 1] - server.y)
        ax_side.plot(dist, tr[:, 2], color=outcome_color(r['outcome']), lw=2.5 if d == 0 else 1.2,
                     label='{:+g} deg -> {}'.format(d, r['outcome']))
    ax_side.plot([net_dist, net_dist], [0, net_height(CENTER_X)], color='black', lw=3)
    ax_side.axvspan(net_dist, net_dist + SERVICE_DEPTH, color='#c8e6c9', alpha=0.4, label='service box depth')
    ax_side.set_xlabel('distance along the court from the server (m)')
    ax_side.set_ylabel('height (m)')
    ax_side.set_title('Side view: launch angle {:.1f} deg +/- changes'.format(base.launch_angle_deg))
    ax_side.set_ylim(bottom=0)
    ax_side.legend(fontsize=8)

    # top view: grid of launch angle x aim, coloured by outcome
    draw_court(ax_top)
    box = server.target_box()
    ax_top.add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], color='#c8e6c9', alpha=0.6))
    grid = [replace(base, launch_angle_deg=base.launch_angle_deg + da, aim_deg=base.aim_deg + aim)
            for da in np.arange(-2, 2.01, 0.5) for aim in range(-6, 7, 2)]
    for r in evaluate(server, grid, model=model):
        ax_top.scatter(r['bounce_x_m'], r['bounce_y_m'], s=18, color=outcome_color(r['outcome']),
                       marker='x' if r['outcome'] == 'net' else 'o')
    base_tr = evaluate(server, [base], model=model, keep_traj=True)[0]['trajectory']
    ax_top.plot(base_tr[:, 0], base_tr[:, 1], color='black', lw=1.5)
    ax_top.scatter([server.x], [server.y], marker='^', s=80, color='black')
    ax_top.set_xlim(-2, COURT_WIDTH + 2)
    ax_top.set_ylim(COURT_LENGTH + 3, -3)
    ax_top.set_aspect('equal')
    ax_top.set_title('Top view: bounce for launch +/-2 deg x aim +/-6 deg')
    for name, color in OUTCOME_COLORS.items():
        ax_top.scatter([], [], color=color, s=18, marker='x' if name == 'net' else 'o',
                       label='net (where it would land)' if name == 'net' else name)
    ax_top.legend(fontsize=7, loc='lower right')

    fig.suptitle('SIMULATION (assumed racquet speed {:.0f} m/s, spin {:+.0f} rpm, contact {:.2f} m) - '
                 'not measured from the video'.format(base.racquet_speed_ms, base.spin_rpm, base.contact_height_m),
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


def strip_traj(obj):
    if isinstance(obj, dict):
        return {k: strip_traj(v) for k, v in obj.items() if k != 'trajectory'}
    if isinstance(obj, list):
        return [strip_traj(v) for v in obj]
    return obj


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Serve what-if simulations for every serve event')
    parser.add_argument('--events_csv', type=str, required=True, help='from hit_detection/classify_events.py')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--racquet_speed', type=float, default=30.0, help='assumed racquet head speed (m/s)')
    parser.add_argument('--spin_rpm', type=float, default=1500.0, help='assumed spin, + topspin / - backspin')
    parser.add_argument('--launch_angle', type=float, default=None,
                        help='assumed launch angle (deg); default: solved per serve')
    parser.add_argument('--contact_height', type=float, default=None,
                        help='override contact height (m); default: video estimate if plausible, else 2.6')
    parser.add_argument('--contact_forward', type=float, default=0.3, help='contact point in front of the feet (m)')
    parser.add_argument('--apparent_cor', type=float, default=0.4, help='racquet-ball apparent restitution')
    parser.add_argument('--min_plausible_height', type=float, default=1.8)
    parser.add_argument('--max_plausible_height', type=float, default=3.5)
    parser.add_argument('--no_plots', action='store_true')
    args = parser.parse_args()

    events = pd.read_csv(args.events_csv)
    serves = events[events['event_type'] == 'serve']
    os.makedirs(args.output_dir, exist_ok=True)

    summary = []
    for ev in serves.itertuples(index=False):
        notes = []
        cx, cy = getattr(ev, 'court_x_m', np.nan), getattr(ev, 'court_y_m', np.nan)
        if pd.isna(cx) or pd.isna(cy):
            cx, cy = 6.5, 24.0
            notes.append('server position unknown (no court calibration): assumed near side, deuce, 0.23 m '
                         'behind the baseline')
        server = Server(x=float(cx), y=float(cy), apparent_cor=args.apparent_cor)

        height = args.contact_height
        est = getattr(ev, 'contact_height_est_m', np.nan)
        if height is None:
            if pd.notna(est) and args.min_plausible_height <= est <= args.max_plausible_height:
                height = float(est)
                notes.append('contact height {:.2f} m estimated from the video (rough)'.format(height))
            else:
                height = 2.6
                notes.append('contact height assumed 2.6 m (video estimate missing or implausible: {})'
                             .format(None if pd.isna(est) else round(float(est), 2)))

        base = ServeParams(racquet_speed_ms=args.racquet_speed, spin_rpm=args.spin_rpm,
                           contact_height_m=height, contact_forward_m=args.contact_forward,
                           launch_angle_deg=args.launch_angle if args.launch_angle is not None else 0.0)
        analysis = analyse_serve(server, base, solve_angle=args.launch_angle is None)
        analysis['event'] = {'frame': int(ev.frame), 'time_s': float(ev.time_s), 'track_id': int(ev.track_id)}
        analysis['notes'] = notes + ['racquet speed, spin and launch angle are assumptions, not measurements']

        name = 'serve_{:06d}'.format(int(ev.frame))
        with open(os.path.join(args.output_dir, name + '.json'), 'w') as f:
            json.dump(strip_traj(analysis), f, indent=2)
        if not args.no_plots:
            plot_serve(os.path.join(args.output_dir, name + '.png'), server, analysis)

        summary.append({'file': name + '.json', **analysis['event'],
                        'server_side': server.side, 'contact_height_m': round(height, 2),
                        'baseline_outcome': analysis['baseline']['outcome'],
                        'launch_window_deg': analysis['launch_windows']['baseline']['width_deg'],
                        'top_suggestion': analysis['suggestions'][0]['text'] if analysis['suggestions'] else None})
        print('{}: {} side, contact {:.2f} m, window {:.1f} deg'.format(
            name, server.side, height, analysis['launch_windows']['baseline']['width_deg']))

    with open(os.path.join(args.output_dir, 'serves_summary.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print('Simulated {} serve events, results in {}'.format(len(summary), args.output_dir))
