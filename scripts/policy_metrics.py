"""Compute benchmark metrics from a single policy_eval recording.

Reads <policy_dir>/manifest.json plus each scenario's 50 Hz and 200 Hz
NPZ pair; prints a markdown table and optionally writes CSV/JSON.

Convention:
  - 200 Hz substep stream (`_200hz.npz`) is the source of truth for foot
    kinematics, contact forces, and base state. Impact metrics and
    body acceleration use it.
  - 50 Hz control stream is used for tracking RMSEs and `action_rate_rms`
    (policy actions are predicted at 50 Hz).
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import tyro


# 12 substeps = 60 ms at 200 Hz. The fz peak can land several substeps after the touchdown firing instant.
_FZ_PEAK_WINDOW = 12


@dataclass
class Args:
    """Compute benchmark metrics from a single policy_eval recording."""

    policy_dir: Path
    """policy_eval/ directory to evaluate."""

    csv_out: Path | None = None
    """write per-scenario CSV here."""

    json_out: Path | None = None
    """write flat JSON here (wandb-ready)."""


def quat_rotate_inverse(quat_xyzw: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate world-frame vector into base frame; quaternion is (x, y, z, w).

    Mirrors utils/rotations.py:quat_rotate_inverse(w_last=True).
    """
    q_vec = quat_xyzw[..., :3]
    q_w = quat_xyzw[..., 3:4]
    a = vec * (2.0 * q_w**2 - 1.0)
    b = np.cross(q_vec, vec, axis=-1) * q_w * 2.0
    c = q_vec * np.sum(q_vec * vec, axis=-1, keepdims=True) * 2.0
    return a - b + c


def detect_touchdowns(vz: np.ndarray, fz: np.ndarray,
                      arm_vz: float = -0.3, fire_fz: float = 1.0) -> np.ndarray:
    """Touchdown indices for one foot.

    Arm when vz < arm_vz; fire on rising edge of fz through fire_fz.
    Same algorithm as utils/touchdown.py:TouchdownDetector.
    """
    events = []
    armed = False
    for i in range(len(vz)):
        if not armed and vz[i] < arm_vz:
            armed = True
        if armed and i > 0 and fz[i - 1] < fire_fz <= fz[i]:
            events.append(i)
            armed = False
    return np.asarray(events, dtype=np.int64)


def vxy_rmse(vel_body_xy: np.ndarray, cmd_xy: np.ndarray) -> float:
    """Body-frame linear velocity tracking RMSE.

    Reference: tracking_lin_vel reward, legged_gym / walk-these-ways
    (Margolis & Agrawal, CoRL 2023, arXiv:2212.03238).
    """
    err = vel_body_xy - cmd_xy
    return float(np.sqrt(np.mean(np.sum(err**2, axis=-1))))


def vyaw_rmse(omega_body_z: np.ndarray, cmd_yaw: np.ndarray) -> float:
    """Yaw rate tracking RMSE.

    Reference: tracking_ang_vel reward, legged_gym.
    """
    return float(np.sqrt(np.mean((omega_body_z - cmd_yaw) ** 2)))


def touchdown_vz_peak(vz_at_td: np.ndarray) -> float:
    """Peak downward foot velocity across touchdown events.

    Reference: walk-these-ways _reward_feet_impact_vel
    (Margolis & Agrawal, arXiv:2212.03238).
    """
    if len(vz_at_td) == 0:
        return 0.0
    return float(np.max(np.abs(np.minimum(vz_at_td, 0.0))))


def touchdown_vz_rms(vz_at_td: np.ndarray) -> float:
    """RMS downward foot velocity across touchdown events.

    Reference: walk-these-ways _reward_feet_impact_vel
    (Margolis & Agrawal, arXiv:2212.03238).
    """
    if len(vz_at_td) == 0:
        return 0.0
    vz_down = np.minimum(vz_at_td, 0.0)
    return float(np.sqrt(np.mean(vz_down**2)))


def fz_peak_at_touchdown_mean(td_idx: np.ndarray, fz: np.ndarray) -> float:
    """Mean peak vertical contact force in a window after each touchdown.

    Reference: QuietWalk (arXiv:2604.23702).
    """
    if len(td_idx) == 0:
        return 0.0
    peaks = np.array([fz[i : i + _FZ_PEAK_WINDOW].max() for i in td_idx])
    return float(np.mean(peaks))


def action_rate_rms(action: np.ndarray, dt: float) -> float:
    """RMS action first-difference per second.

    Reference: action_rate_l2 reward, legged_gym.
    """
    da_per_s = np.diff(action, axis=0) / dt
    return float(np.sqrt(np.mean(np.sum(da_per_s**2, axis=-1))))


def body_accel_rms(lin_vel_world: np.ndarray, dt: float) -> float:
    """RMS base linear acceleration magnitude (finite-diff of base velocity)."""
    a = np.diff(lin_vel_world, axis=0) / dt
    return float(np.sqrt(np.mean(np.sum(a**2, axis=-1))))


def _impact_metrics(foot_vel: np.ndarray, foot_force: np.ndarray,
                    is_active: np.ndarray) -> dict[str, float]:
    """Touchdown-derived metrics. Events firing outside is_active are dropped."""
    td_left = detect_touchdowns(foot_vel[:, 0, 2], foot_force[:, 0, 2])
    td_right = detect_touchdowns(foot_vel[:, 1, 2], foot_force[:, 1, 2])
    td_left = td_left[is_active[td_left]]
    td_right = td_right[is_active[td_right]]
    # Approach velocity span the threshold crossing: take the more
    # downward of {pre, post} samples; whichever is pre-impulse holds the
    # actual impact speed regardless of when within step i-1 -> i the
    # solver applied the contact impulse.
    vz_at_td = np.concatenate([
        np.minimum(foot_vel[td_left - 1, 0, 2], foot_vel[td_left, 0, 2]),
        np.minimum(foot_vel[td_right - 1, 1, 2], foot_vel[td_right, 1, 2]),
    ])
    fz_left, fz_right = foot_force[:, 0, 2], foot_force[:, 1, 2]
    return {
        "touchdown_vz_peak": touchdown_vz_peak(vz_at_td),
        "touchdown_vz_rms": touchdown_vz_rms(vz_at_td),
        "fz_peak_at_touchdown_mean": 0.5 * (
            fz_peak_at_touchdown_mean(td_left, fz_left)
            + fz_peak_at_touchdown_mean(td_right, fz_right)
        ),
    }


def compute_scenario_metrics(npz: dict, npz_hi: dict,
                             dt: float, sim_dt: float) -> dict[str, float]:
    """All metrics on the active window of one scenario."""
    active = npz["is_active"]
    base_quat_lo = npz["base_quat"][active]
    base_lin_vel_lo = npz["base_lin_vel"][active]
    base_ang_vel_lo = npz["base_ang_vel"][active]
    cmd = npz["cmd"][active]
    action = npz["action"][active]

    vel_body = quat_rotate_inverse(base_quat_lo, base_lin_vel_lo)
    omega_body = quat_rotate_inverse(base_quat_lo, base_ang_vel_lo)

    active_hi = npz_hi["is_active"]
    impact = _impact_metrics(npz_hi["foot_vel"], npz_hi["foot_force"], active_hi)

    return {
        "vxy_rmse": vxy_rmse(vel_body[:, :2], cmd[:, :2]),
        "vyaw_rmse": vyaw_rmse(omega_body[:, 2], cmd[:, 2]),
        **impact,
        "action_rate_rms": action_rate_rms(action, dt),
        "body_accel_rms": body_accel_rms(npz_hi["base_lin_vel"][active_hi], sim_dt),
    }


def load_policy_dir(policy_dir: Path) -> tuple[dict, list[tuple[str, dict, dict]]]:
    """Load manifest and (50 Hz, 200 Hz) NPZ pairs from a policy_eval directory."""
    manifest = json.loads((policy_dir / "manifest.json").read_text())
    scenarios = []
    for entry in manifest["scenarios"]:
        stem = f"scenario_{entry['index']:02d}_{entry['name']}"
        lo = dict(np.load(policy_dir / f"{stem}.npz"))
        hi = dict(np.load(policy_dir / f"{stem}_200hz.npz"))
        scenarios.append((entry["name"], lo, hi))
    return manifest, scenarios


def render_markdown(rows: list[dict], metric_names: list[str]) -> str:
    headers = ["scenario"] + metric_names
    lines = ["| " + " | ".join(headers) + " |",
             "|" + "|".join(["---"] * len(headers)) + "|"]
    for r in rows:
        lines.append("| " + " | ".join(
            [r["scenario"]] + [f"{r[m]:.4f}" for m in metric_names]
        ) + " |")
    return "\n".join(lines)


def main(args: Args) -> None:
    manifest, scenarios = load_policy_dir(args.policy_dir)
    dt = 1.0 / manifest["policy_hz"]
    sim_dt = 1.0 / manifest["sim_hz"]

    print(f"# {args.policy_dir.name}")

    rows = [{"scenario": name, **compute_scenario_metrics(lo, hi, dt, sim_dt)}
            for name, lo, hi in scenarios]
    metric_names = [k for k in rows[0].keys() if k != "scenario"]

    avg = {"scenario": "avg",
           **{m: float(np.mean([r[m] for r in rows])) for m in metric_names}}
    rows.append(avg)

    print(render_markdown(rows, metric_names))

    if args.csv_out:
        with args.csv_out.open("w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["scenario"] + metric_names)
            w.writeheader()
            w.writerows(rows)
        print(f"\nwrote {args.csv_out}")

    if args.json_out:
        flat = {f"{r['scenario']}/{m}": r[m] for r in rows for m in metric_names}
        args.json_out.write_text(json.dumps(flat, indent=2))
        print(f"wrote {args.json_out}")


if __name__ == "__main__":
    main(tyro.cli(Args))
