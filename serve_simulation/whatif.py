"""Compare a serve before and after changing its parameters (for an interactive panel).

Prints JSON with both simulated serves: outcome, bounce point, net clearance and a
down-sampled trajectory (court metres) the frontend can draw.

Example:
  python whatif.py --server_x 6.5 --server_y 24.0 --before '{"racquet_speed_ms": 30, "spin_rpm": 0}' \
      --after '{"spin_rpm": 2000}'
`--after` only lists the parameters that change. Without `launch_angle_deg` in --before,
the launch angle is solved so the "before" serve lands about 1 m inside the service line.
"""
import argparse
import json
from dataclasses import replace

from serve import Server, ServeParams, evaluate, launch_window, solve_launch_angle


def compare(server, before, after, points=60):
    out = {}
    for name, params in (('before', before), ('after', after)):
        res = evaluate(server, [params], keep_traj=True)[0]
        tr = res.pop('trajectory')
        step = max(1, len(tr) // points)
        res['trajectory'] = [[round(float(v), 3) for v in p] for p in list(tr[::step]) + [tr[-1]]]
        res['launch_window'] = launch_window(server, params)
        out[name] = res
    out['label'] = 'SIMULATION under assumed parameters, not a measurement'
    return out


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Before/after serve simulation')
    parser.add_argument('--server_x', type=float, default=6.5, help='server court x (m)')
    parser.add_argument('--server_y', type=float, default=24.0, help='server court y (m)')
    parser.add_argument('--before', type=str, default='{}', help='JSON of ServeParams fields')
    parser.add_argument('--after', type=str, default='{}', help='JSON of the ServeParams fields to change')
    parser.add_argument('--apparent_cor', type=float, default=0.4)
    args = parser.parse_args()

    server = Server(x=args.server_x, y=args.server_y, apparent_cor=args.apparent_cor)
    before_fields = json.loads(args.before)
    before = ServeParams(**before_fields)
    if 'launch_angle_deg' not in before_fields:
        before = replace(before, launch_angle_deg=solve_launch_angle(server, before)[0])
    after = replace(before, **json.loads(args.after))
    print(json.dumps(compare(server, before, after)))
