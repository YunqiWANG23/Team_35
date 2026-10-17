"""Serve what-if model on top of physics.py.

Court frame (same as court_calibration/export_homography.py), metres:
  x: 0 .. 10.97 across the court (camera left -> right)
  y: 0 (far baseline) .. 23.77 (near baseline), net at y = 11.885
  z: height above the ground
This frame is left-handed, so physics runs in (x, -y, z) and results are mapped back.

Adjustable serve parameters (the same set as TopSpin Ball Lab's sliders, plus where
the ball is struck relative to the server's feet):
  racquet_speed_ms   racquet head speed at contact
  launch_angle_deg   vertical launch angle (+ up, - down)
  aim_deg            horizontal angle from the line to the target box centre (+ = server's right)
  spin_rpm           + topspin / - backspin
  contact_height_m   ball height at contact
  contact_forward_m  how far in front of the server's feet the ball is struck

Ball speed after contact uses a simple impact model for a ball that is nearly at rest
(the tossed serve):  v_ball = (1 + e_A) * v_racquet, with apparent coefficient of
restitution e_A (default 0.4). All outputs are SIMULATIONS under these assumptions.
"""
from dataclasses import dataclass, asdict, replace

import numpy as np

from physics import BallModel, simulate

COURT_WIDTH, COURT_LENGTH = 10.97, 23.77
NET_Y = COURT_LENGTH / 2
SINGLES_LEFT, SINGLES_RIGHT = 1.37, 9.60
CENTER_X = COURT_WIDTH / 2
SERVICE_DEPTH = 6.40                  # net to service line
NET_HEIGHT_CENTER, NET_HEIGHT_POST = 0.914, 1.07
NET_POST_LEFT, NET_POST_RIGHT = SINGLES_LEFT - 0.914, SINGLES_RIGHT + 0.914


@dataclass
class ServeParams:
    racquet_speed_ms: float = 30.0
    launch_angle_deg: float = 0.0
    aim_deg: float = 0.0
    spin_rpm: float = 1500.0
    contact_height_m: float = 2.6
    contact_forward_m: float = 0.3


@dataclass
class Server:
    x: float = 6.5      # court metres (ground position of the server)
    y: float = 24.0
    apparent_cor: float = 0.4

    @property
    def side(self):
        return 'near' if self.y > NET_Y else 'far'

    @property
    def forward(self):
        """ Unit direction towards the net in court coordinates """
        return np.array([0.0, -1.0]) if self.side == 'near' else np.array([0.0, 1.0])

    def target_box(self):
        """ Diagonal service box: (x_min, x_max, y_min, y_max) """
        if self.x >= CENTER_X:
            x_range = (SINGLES_LEFT, CENTER_X)
        else:
            x_range = (CENTER_X, SINGLES_RIGHT)
        if self.side == 'near':
            y_range = (NET_Y - SERVICE_DEPTH, NET_Y)
        else:
            y_range = (NET_Y, NET_Y + SERVICE_DEPTH)
        return x_range + y_range


def net_height(x):
    x = np.clip(x, NET_POST_LEFT, NET_POST_RIGHT)
    half = np.where(x < CENTER_X, CENTER_X - NET_POST_LEFT, NET_POST_RIGHT - CENTER_X)
    return NET_HEIGHT_CENTER + (NET_HEIGHT_POST - NET_HEIGHT_CENTER) * np.abs(x - CENTER_X) / half


def _to_world(p):
    p = np.array(p, dtype=float)
    p[..., 1] *= -1
    return p


def initial_state(server, params_list):
    """ Court-frame contact point -> world-frame position, velocity, spin for each ServeParams """
    pos, vel, spin = [], [], []
    box = server.target_box()
    target = np.array([(box[0] + box[1]) / 2, (box[2] + box[3]) / 2])
    for p in params_list:
        fwd = server.forward
        contact = np.array([server.x, server.y]) + fwd * p.contact_forward_m
        to_target = target - contact
        d = _to_world(np.append(to_target / np.linalg.norm(to_target), 0.0))
        up = np.array([0.0, 0.0, 1.0])
        right = np.cross(d, up)
        a = np.radians(p.aim_deg)
        d = np.cos(a) * d + np.sin(a) * right
        el = np.radians(p.launch_angle_deg)
        speed = (1 + server.apparent_cor) * p.racquet_speed_ms
        vel.append(speed * (np.cos(el) * d + np.sin(el) * up))
        # topspin rotates the top of the ball forward: axis points to the left of travel
        spin.append(p.spin_rpm * 2 * np.pi / 60 * np.cross(up, d))
        pos.append(_to_world([contact[0], contact[1], p.contact_height_m]))
    return np.array(pos), np.array(vel), np.array(spin)


def evaluate(server, params_list, model=None, keep_traj=False, dt=0.002):
    """ Simulate serves and score each one against the target service box """
    pos, vel, spin = initial_state(server, params_list)
    sim = simulate(pos, vel, spin, model=model, dt=dt)
    traj = sim['traj'].copy()
    traj[..., 1] *= -1  # back to court frame
    radius = (model or BallModel()).radius
    box = server.target_box()
    results = []
    for i, p in enumerate(params_list):
        tr = traj[i][~np.isnan(traj[i, :, 0])]
        res = {'params': asdict(p),
               'ball_speed_kmh': round((1 + server.apparent_cor) * p.racquet_speed_ms * 3.6, 1),
               'peak_height_m': round(float(tr[:, 2].max()), 3)}

        rel = (tr[:, 1] - NET_Y) * (1 if server.side == 'far' else -1)  # > 0 once past the net
        crossed = np.where((rel[:-1] < 0) & (rel[1:] >= 0))[0]
        if len(crossed):
            j = crossed[0]
            w = -rel[j] / (rel[j + 1] - rel[j])
            pt = tr[j] + w * (tr[j + 1] - tr[j])
            res['net_clearance_m'] = round(float(pt[2] - radius - net_height(pt[0])), 3)
        else:
            res['net_clearance_m'] = None

        bounce = tr[-1]
        res['bounce_x_m'], res['bounce_y_m'] = round(float(bounce[0]), 3), round(float(bounce[1]), 3)
        res['flight_time_s'] = round(float((len(tr) - 1) * dt), 3)
        if len(tr) > 1:
            res['bounce_speed_kmh'] = round(float(np.linalg.norm(tr[-1] - tr[-2]) / dt * 3.6), 1)

        # signed distance to the box edges: > 0 inside
        inside_x = min(bounce[0] - box[0], box[1] - bounce[0])
        inside_y = min(bounce[1] - box[2], box[3] - bounce[1])
        if res['net_clearance_m'] is None or res['net_clearance_m'] < 0 or not sim['landed'][i]:
            outcome = 'net'
        elif inside_x >= 0 and inside_y >= 0:
            outcome = 'in'
        else:
            past_line = abs(bounce[1] - NET_Y) > SERVICE_DEPTH
            outer = SINGLES_LEFT if box[0] == SINGLES_LEFT else SINGLES_RIGHT
            off_outer = abs(bounce[0] - CENTER_X) > abs(outer - CENTER_X)
            parts = (['long'] if past_line else []) + (['wide'] if off_outer else []) + \
                    (['centre'] if not off_outer and inside_x < 0 else [])
            outcome = '_'.join(parts) or 'out'
        res['outcome'] = outcome
        res['box_margin_m'] = round(float(min(inside_x, inside_y)), 3)
        res['depth_from_net_m'] = round(float(abs(bounce[1] - NET_Y)), 3)
        if keep_traj:
            res['trajectory'] = tr
        results.append(res)
    return results


def launch_window(server, params, model=None, lo=-15.0, hi=15.0, step=0.1):
    """ Range of launch angles (other parameters fixed) whose serve lands in the box """
    angles = np.round(np.arange(lo, hi + step / 2, step), 3)
    res = evaluate(server, [replace(params, launch_angle_deg=float(a)) for a in angles], model=model, dt=0.004)
    ok = [a for a, r in zip(angles, res) if r['outcome'] == 'in']
    if not ok:
        return {'width_deg': 0.0, 'min_deg': None, 'max_deg': None}
    return {'width_deg': round(float(ok[-1] - ok[0] + step), 2), 'min_deg': float(ok[0]), 'max_deg': float(ok[-1])}


def solve_launch_angle(server, params, model=None, target_inside_m=1.0):
    """ Launch angle that lands target_inside_m in front of the service line (or the best try) """
    angles = np.round(np.arange(-15.0, 15.05, 0.1), 2)
    res = evaluate(server, [replace(params, launch_angle_deg=float(a)) for a in angles], model=model, dt=0.004)
    goal = SERVICE_DEPTH - target_inside_m
    ins = [(abs(r['depth_from_net_m'] - goal), a) for a, r in zip(angles, res) if r['outcome'] == 'in']
    if ins:
        return float(min(ins)[1]), True
    cleared = [(abs(r['depth_from_net_m'] - goal), a) for a, r in zip(angles, res) if r['outcome'] != 'net']
    if cleared:
        return float(min(cleared)[1]), False
    return 0.0, False


SWEEPS = {
    'launch_angle_deg': [-2, -1, -0.5, 0.5, 1, 2],
    'aim_deg': [-6, -4, -2, 2, 4, 6],
    'racquet_speed_ms': [-5, -2.5, 2.5, 5],
    'spin_rpm': [-1500, -1000, -500, 500, 1000, 1500],
    'contact_height_m': [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3],
    'contact_forward_m': [-0.3, 0.3, 0.6],
}
WINDOW_SWEEPS = {
    'contact_height_m': [-0.3, -0.15, 0.15, 0.3],
    'spin_rpm': [-1000, 1000, 2000],
    'racquet_speed_ms': [-5, 5],
}


def sweep(server, base, model=None):
    """ One-at-a-time changes around the baseline """
    out = {}
    for name, deltas in SWEEPS.items():
        variants = [replace(base, **{name: getattr(base, name) + d}) for d in deltas]
        out[name] = [dict(delta=d, **{k: v for k, v in r.items() if k != 'params'},
                          value=round(getattr(p, name), 3))
                     for d, p, r in zip(deltas, variants, evaluate(server, variants, model=model))]
    return out


def window_sweep(server, base, model=None):
    """ How forgiving the serve is (launch-angle window) as other parameters change """
    out = {'baseline': launch_window(server, base, model)}
    for name, deltas in WINDOW_SWEEPS.items():
        out[name] = []
        for d in deltas:
            p = replace(base, **{name: getattr(base, name) + d})
            out[name].append(dict(delta=d, value=round(getattr(p, name), 3), **launch_window(server, p, model)))
    return out


LABELS = {
    'contact_height_m': ('contact height', 'm'),
    'spin_rpm': ('spin', 'rpm'),
    'racquet_speed_ms': ('racquet speed', 'm/s'),
    'launch_angle_deg': ('launch angle', 'deg'),
    'aim_deg': ('aim', 'deg'),
    'contact_forward_m': ('contact point in front of the feet', 'm'),
}


def suggestions(base_result, sweeps, windows):
    """ Rule-based, simulation-only suggestions, most useful first """
    tips = []
    base_win = windows['baseline']['width_deg']

    if base_result['outcome'] != 'in':
        fixes = []
        for name, rows in sweeps.items():
            for r in rows:
                if r['outcome'] == 'in':
                    fixes.append((abs(r['delta']) / {'launch_angle_deg': 1, 'aim_deg': 2, 'racquet_speed_ms': 2.5,
                                                     'spin_rpm': 500, 'contact_height_m': 0.1,
                                                     'contact_forward_m': 0.3}[name], name, r))
        for _, name, r in sorted(fixes, key=lambda f: f[0])[:3]:
            label, unit = LABELS[name]
            tips.append({'parameter': name, 'change': r['delta'], 'kind': 'fix',
                         'text': 'Simulated serve is {}. Changing {} by {:+g} {} lands it in the box '
                                 '(margin {:.2f} m).'.format(base_result['outcome'], label, r['delta'], unit,
                                                             r['box_margin_m'])})

    for name in ('contact_height_m', 'spin_rpm', 'racquet_speed_ms'):
        label, unit = LABELS[name]
        best = max(windows[name], key=lambda r: r['width_deg'])
        gain = best['width_deg'] - base_win
        if gain >= 0.3:
            tips.append({'parameter': name, 'change': best['delta'], 'kind': 'tolerance',
                         'window_before_deg': base_win, 'window_after_deg': best['width_deg'],
                         'text': 'Changing {} by {:+g} {} (to {:g} {}) widens the range of launch angles that '
                                 'land in the box from {:.1f} deg to {:.1f} deg, so the serve is more '
                                 'forgiving.'.format(label, best['delta'], unit, round(best['value'], 2), unit,
                                                     base_win, best['width_deg'])})
            if name == 'racquet_speed_ms' and best['delta'] < 0:
                base_speed = base_result['params']['racquet_speed_ms']
                new_kmh = base_result['ball_speed_kmh'] * best['value'] / base_speed
                tips[-1]['text'] += ' Trade-off: ball speed drops from {:.0f} to {:.0f} km/h.'.format(
                    base_result['ball_speed_kmh'], new_kmh)

    one = [r for r in sweeps['launch_angle_deg'] if abs(r['delta']) == 1]
    if len(one) == 2:
        shift = abs(one[0]['depth_from_net_m'] - one[1]['depth_from_net_m']) / 2
        tips.append({'parameter': 'launch_angle_deg', 'change': 1, 'kind': 'sensitivity',
                     'text': 'A 1 deg change in launch angle moves the bounce about {:.2f} m deeper or shorter, '
                             'with a total in-box window of {:.1f} deg.'.format(shift, base_win)})

    aims_in = [r['delta'] for r in sweeps['aim_deg'] if r['outcome'] == 'in']
    if aims_in:
        tips.append({'parameter': 'aim_deg', 'change': None, 'kind': 'sensitivity',
                     'text': 'Aim changes between {:+g} and {:+g} deg (server\'s right is +) still land in the '
                             'box.'.format(min(aims_in + [0]), max(aims_in + [0]))})
    for t in tips:
        t['text'] = 'Simulation: ' + t['text']
    return tips


def analyse_serve(server, base, model=None, solve_angle=True):
    """ Full what-if analysis for one serve event """
    solved = None
    if solve_angle:
        angle, solved = solve_launch_angle(server, base, model)
        base = replace(base, launch_angle_deg=angle)
    base_result = evaluate(server, [base], model=model)[0]
    sw = sweep(server, base, model)
    win = window_sweep(server, base, model)
    return {
        'server': {'x_m': server.x, 'y_m': server.y, 'side': server.side,
                   'target_box': dict(zip(['x_min', 'x_max', 'y_min', 'y_max'], server.target_box()))},
        'baseline': base_result,
        'baseline_launch_angle_solved': solved,
        'sweeps': sw,
        'launch_windows': win,
        'suggestions': suggestions(base_result, sw, win),
    }
