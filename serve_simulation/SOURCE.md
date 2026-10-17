physics.py ports the flight model (drag, Magnus, spin decay, RK4) from
https://github.com/orcn-dev/tennis-physics-3d-demo (tennis_cli/index.js, commit 75af4a3)
by orcn-dev to Python/NumPy, with the changes listed at the top of physics.py.
That repository has no license file; its README says "Fork, try, tweak".

The adjustable serve parameters (racquet speed, launch angle, aim, spin +top/-back,
contact height) follow the sliders of TopSpin Ball Lab
(https://topspin-labs.vercel.app/ball-lab). No code, model or API from that site is used.

serve.py, simulate_serves.py and whatif.py are our own code.
