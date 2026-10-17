"""Tennis ball flight model: gravity + aerodynamic drag + Magnus lift, spin decay, RK4.

Ported to Python/NumPy from the JavaScript model in
https://github.com/orcn-dev/tennis-physics-3d-demo (tennis_cli/index.js):
  drag    F_d = -1/2 rho A C_D |v| v
  Magnus  F_m = 1/2 rho A C_L |v| (w_hat x v)
  spin    w(t) = w0 * exp(-t / tau)
  RK4 integration with a fixed time step.

Changes from the demo:
  - many shots are integrated at once (vectorised over N shots);
  - spin is passed as a physical angular velocity vector (rad/s) in a right-handed frame,
    so the sign of the Magnus force follows the real rotation;
  - besides the demo lift law (C_L = CL0 * min(1.2, S)), the default lift law is the
    widely used empirical fit C_L = 1 / (2 + 1/S), S = R|w|/|v|, with C_D = 0.55;
  - wind is a constant vector (no gusts) and the flight stops at the first bounce.

All quantities are SI units. The frame must be right-handed with z pointing up.
"""
from dataclasses import dataclass

import numpy as np


@dataclass
class BallModel:
    mass: float = 0.0577         # kg
    radius: float = 0.0335       # m
    air_density: float = 1.21    # kg/m^3
    drag_coeff: float = 0.55
    lift_model: str = 'empirical'  # 'empirical' (1 / (2 + 1/S)) or 'demo' (CL0 * min(1.2, S))
    lift_coeff_demo: float = 0.20  # CL0 for the 'demo' lift model
    spin_decay_s: float = 4.0    # tau
    gravity: float = 9.81
    wind: tuple = (0.0, 0.0, 0.0)

    @property
    def area(self):
        return np.pi * self.radius ** 2


def _accel(vel, spin, model):
    """ Acceleration for N balls: vel (N,3), spin (N,3) rad/s """
    v_rel = vel - np.asarray(model.wind, dtype=float)
    speed = np.linalg.norm(v_rel, axis=1, keepdims=True)
    speed_safe = np.maximum(speed, 1e-6)
    k = 0.5 * model.air_density * model.area / model.mass

    acc = -k * model.drag_coeff * speed * v_rel

    w = np.linalg.norm(spin, axis=1, keepdims=True)
    has_spin = w[:, 0] > 1e-8
    if has_spin.any():
        S = model.radius * w / speed_safe
        if model.lift_model == 'demo':
            cl = model.lift_coeff_demo * np.minimum(1.2, S)
        else:
            cl = 1.0 / (2.0 + 1.0 / np.maximum(S, 1e-9))
        w_hat = np.where(has_spin[:, None], spin / np.maximum(w, 1e-12), 0.0)
        acc = acc + k * cl * speed * np.cross(w_hat, v_rel)

    acc[:, 2] -= model.gravity
    return acc


def simulate(pos0, vel0, spin0, model=None, dt=0.002, t_max=3.0):
    """ Integrate N shots until each one first touches the ground (z <= radius)

    :params
        pos0, vel0: (N,3) initial position (m) and velocity (m/s)
        spin0: (N,3) initial angular velocity (rad/s)
    :return dict with
        traj: (N, steps, 3) positions, NaN after the ball has landed
        t: (steps,) time stamps
        landed: (N,) bool
    """
    model = model or BallModel()
    pos = np.array(pos0, dtype=float).reshape(-1, 3)
    vel = np.array(vel0, dtype=float).reshape(-1, 3)
    spin0 = np.array(spin0, dtype=float).reshape(-1, 3)
    n = len(pos)
    steps = int(t_max / dt) + 1

    traj = np.full((n, steps, 3), np.nan)
    traj[:, 0] = pos
    active = pos[:, 2] > model.radius
    landed = ~active
    t = 0.0
    last = 0
    for i in range(1, steps):
        if not active.any():
            break
        decay = lambda tt: spin0 * np.exp(-tt / model.spin_decay_s)
        p, v = pos[active], vel[active]
        s1, s2, s4 = decay(t)[active], decay(t + dt / 2)[active], decay(t + dt)[active]

        k1v = _accel(v, s1, model);                 k1p = v
        k2v = _accel(v + k1v * dt / 2, s2, model);  k2p = v + k1v * dt / 2
        k3v = _accel(v + k2v * dt / 2, s2, model);  k3p = v + k2v * dt / 2
        k4v = _accel(v + k3v * dt, s4, model);      k4p = v + k3v * dt

        pos[active] = p + dt / 6 * (k1p + 2 * k2p + 2 * k3p + k4p)
        vel[active] = v + dt / 6 * (k1v + 2 * k2v + 2 * k3v + k4v)
        t += dt

        traj[active, i] = pos[active]
        touched = active & (pos[:, 2] <= model.radius)
        landed |= touched
        active &= ~touched
        last = i
    return {'traj': traj[:, :last + 1], 't': np.arange(last + 1) * dt, 'landed': landed}
