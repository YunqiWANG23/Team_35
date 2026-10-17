"""Run the full pipeline on one uploaded video.

Part 1 - recognition:  court homography -> players -> ball -> hits -> events
Part 2 - suggestion:   serve what-if simulations for every serve event

Each step is a separate script in its module folder; this file only runs them in
order (with the module folder as working directory) and passes the files along.

Example:
  python run_pipeline.py --video videos/match.mp4 --output_dir output/match \
      --court_model court_calibration/weights/court_model.pt \
      --ball_model ball_tracking/weights/ball_model.pt
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))


def run(step, module, script, *script_args, skip_if=None, skip_existing=False):
    if skip_existing and skip_if and os.path.exists(skip_if):
        print('[{}] skipped, {} exists'.format(step, skip_if))
        return
    print('[{}] {} {}'.format(step, script, ' '.join(script_args)))
    subprocess.run([sys.executable, script, *script_args], cwd=os.path.join(ROOT, module), check=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Video -> events -> serve simulations')
    parser.add_argument('--video', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--court_model', type=str, required=True)
    parser.add_argument('--ball_model', type=str, required=True)
    parser.add_argument('--player_model', type=str, default='yolo11s-pose.pt')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda', 'mps'])
    parser.add_argument('--racquet_speed', type=float, default=30.0, help='assumed racquet speed for simulations')
    parser.add_argument('--spin_rpm', type=float, default=1500.0, help='assumed spin for simulations')
    parser.add_argument('--player_height', type=float, default=1.80, help='used for the contact height estimate')
    parser.add_argument('--skip_existing', action='store_true', help='skip steps whose output already exists')
    args = parser.parse_args()

    a = lambda p: os.path.abspath(p)
    out = a(args.output_dir)
    os.makedirs(out, exist_ok=True)
    f = lambda name: os.path.join(out, name)
    video = a(args.video)
    yolo_device = [] if args.device == 'auto' else ['--device', '0' if args.device == 'cuda' else args.device]
    skip = dict(skip_existing=args.skip_existing)

    run('1/6 court', 'court_calibration', 'export_homography.py', '--model_path', a(args.court_model),
        '--input_path', video, '--output_json', f('court.json'), '--device', args.device,
        skip_if=f('court.json'), **skip)
    run('2/6 players', 'player_tracking', 'track_players.py', '--input_path', video,
        '--output_csv', f('players.csv'), '--court_homography', f('court.json'), '--players_only',
        '--model', args.player_model, *yolo_device, skip_if=f('players.csv'), **skip)
    run('3/6 ball', 'ball_tracking', 'track_ball.py', '--model_path', a(args.ball_model), '--input_path', video,
        '--output_csv', f('ball.csv'), '--interpolate', '--device', args.device, skip_if=f('ball.csv'), **skip)
    run('4/6 hits', 'hit_detection', 'detect_hits.py', '--ball_csv', f('ball.csv'), '--players_csv',
        f('players.csv'), '--output_csv', f('hits.csv'), '--summary_json', f('hits_summary.json'),
        skip_if=f('hits.csv'), **skip)
    run('5/6 events', 'hit_detection', 'classify_events.py', '--hits_csv', f('hits.csv'), '--players_csv',
        f('players.csv'), '--output_csv', f('events.csv'), '--player_height', str(args.player_height),
        skip_if=f('events.csv'), **skip)
    run('6/6 serve simulation', 'serve_simulation', 'simulate_serves.py', '--events_csv', f('events.csv'),
        '--output_dir', f('serves'), '--racquet_speed', str(args.racquet_speed), '--spin_rpm', str(args.spin_rpm))
    print('Done. Results in {}'.format(out))
