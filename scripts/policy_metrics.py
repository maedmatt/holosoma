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

# Baseline for the delta-vs-baseline heatmap. fastsac, symmetry off, flat terrain.
# Source: https://wandb.ai/matteo-calabria01-maedmatt/IDSIA/runs/60zwuqkq
_BASELINE_NAME = "fastsac_no_symm_flat_60zwuqkq"
_BASELINE = {
    "forward_03":   {"vxy_rmse": 0.1722, "vyaw_rmse": 0.1332, "touchdown_vz_peak": 0.8926, "touchdown_vz_rms": 0.8666, "fz_peak_at_touchdown_mean": 1007.82, "action_rate_rms": 21.3010, "body_accel_rms": 3.8982},
    "forward_06":   {"vxy_rmse": 0.1964, "vyaw_rmse": 0.1349, "touchdown_vz_peak": 0.9591, "touchdown_vz_rms": 0.8744, "fz_peak_at_touchdown_mean": 1017.78, "action_rate_rms": 22.4356, "body_accel_rms": 4.6619},
    "forward_09":   {"vxy_rmse": 0.2712, "vyaw_rmse": 0.1771, "touchdown_vz_peak": 1.1099, "touchdown_vz_rms": 0.9448, "fz_peak_at_touchdown_mean": 1024.33, "action_rate_rms": 24.7324, "body_accel_rms": 5.5070},
    "backward_03":  {"vxy_rmse": 0.1843, "vyaw_rmse": 0.1881, "touchdown_vz_peak": 1.0404, "touchdown_vz_rms": 0.9740, "fz_peak_at_touchdown_mean": 1300.63, "action_rate_rms": 21.2034, "body_accel_rms": 3.8724},
    "backward_06":  {"vxy_rmse": 0.2465, "vyaw_rmse": 0.1802, "touchdown_vz_peak": 1.1827, "touchdown_vz_rms": 1.0797, "fz_peak_at_touchdown_mean": 1367.51, "action_rate_rms": 23.1583, "body_accel_rms": 4.1011},
    "backward_09":  {"vxy_rmse": 0.3810, "vyaw_rmse": 0.1311, "touchdown_vz_peak": 1.2760, "touchdown_vz_rms": 1.2109, "fz_peak_at_touchdown_mean": 1499.52, "action_rate_rms": 25.6915, "body_accel_rms": 4.4746},
    "strafe_left":  {"vxy_rmse": 0.2338, "vyaw_rmse": 0.1355, "touchdown_vz_peak": 0.9401, "touchdown_vz_rms": 0.9175, "fz_peak_at_touchdown_mean": 1107.19, "action_rate_rms": 21.1900, "body_accel_rms": 3.9158},
    "strafe_right": {"vxy_rmse": 0.2219, "vyaw_rmse": 0.1365, "touchdown_vz_peak": 1.0245, "touchdown_vz_rms": 0.9088, "fz_peak_at_touchdown_mean": 1100.79, "action_rate_rms": 21.4351, "body_accel_rms": 3.8061},
    "yaw_left":     {"vxy_rmse": 0.1677, "vyaw_rmse": 0.1266, "touchdown_vz_peak": 0.9460, "touchdown_vz_rms": 0.8656, "fz_peak_at_touchdown_mean": 1003.12, "action_rate_rms": 21.0539, "body_accel_rms": 3.6745},
    "yaw_right":    {"vxy_rmse": 0.1725, "vyaw_rmse": 0.2016, "touchdown_vz_peak": 0.9490, "touchdown_vz_rms": 0.9104, "fz_peak_at_touchdown_mean": 1081.99, "action_rate_rms": 21.1834, "body_accel_rms": 3.7517},
    "avg":          {"vxy_rmse": 0.2247, "vyaw_rmse": 0.1545, "touchdown_vz_peak": 1.0320, "touchdown_vz_rms": 0.9553, "fz_peak_at_touchdown_mean": 1151.07, "action_rate_rms": 22.3385, "body_accel_rms": 4.1663},
}

# Per-metric scenario mask. A metric appears on the delta heatmap only where the scenario
# actively exercises it. Metrics that measure drift around zero (vyaw_rmse on non-yaw
# scenarios, vxy_rmse on yaw scenarios) get masked because tiny absolute drifts produce
# large percent deltas when the baseline value is near zero. Metrics not listed
# here are relevant on every scenario.
_RELEVANT_SCENARIOS = {
    "vxy_rmse":  {"forward_03", "forward_06", "forward_09",
                  "backward_03", "backward_06", "backward_09",
                  "strafe_left", "strafe_right"},
    "vyaw_rmse": {"yaw_left", "yaw_right"},
}


def _is_relevant(scenario: str, metric: str) -> bool:
    if scenario == "avg":
        return True
    relevant = _RELEVANT_SCENARIOS.get(metric)
    return relevant is None or scenario in relevant


@dataclass
class Args:
    """Compute benchmark metrics from a single policy_eval recording."""

    policy_dir: Path
    """policy_eval/ directory to evaluate."""

    csv_out: Path | None = None
    """write per-scenario CSV here."""

    json_out: Path | None = None
    """write flat JSON here (wandb-ready)."""

    resume_from_ckpt: Path | None = None
    """checkpoint .pt that holds the wandb run path; resume that run and log delta-vs-baseline."""


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
        "touchdown_count_left": float(len(td_left)),
        "touchdown_count_right": float(len(td_right)),
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


def _log_to_wandb(rows: list[dict], metric_names: list[str],
                  ckpt_path: Path, policy_dir: Path) -> None:
    """Resume the wandb run named in the checkpoint metadata and log delta-vs-baseline."""
    import plotly.express as px
    import torch
    import wandb

    state = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    if "wandb_run_path" not in state:
        raise ValueError(f"{ckpt_path} has no wandb_run_path metadata.")
    entity, project, run_id = state["wandb_run_path"].split("/")
    # dir= keeps the resumed run's staging files under the eval dir, not cwd.
    # x_disable_meta=True prevents the resume from overwriting training's argv/program.
    wandb.init(
        entity=entity, project=project, id=run_id, resume="allow",
        dir=policy_dir, settings=wandb.Settings(x_disable_meta=True),
    )

    # Touchdown counts are sanity-check diagnostics, not delta metrics — they
    # have no baseline, so they'd render as blank columns in the heatmap.
    heatmap_metrics = [m for m in metric_names if not m.startswith("touchdown_count")]

    scens = [r["scenario"] for r in rows]
    deltas = np.full((len(scens), len(heatmap_metrics)), np.nan)
    for i, scen in enumerate(scens):
        for j, m in enumerate(heatmap_metrics):
            if not _is_relevant(scen, m):
                continue
            base = _BASELINE.get(scen, {}).get(m)
            if base:
                deltas[i, j] = (rows[i][m] - base) / base * 100.0

    # Plotly figure -> wandb logs as interactive chart panel (Charts section, not Media).
    vbound = float(np.nanmax(np.abs(deltas))) if not np.all(np.isnan(deltas)) else 50.0
    fig = px.imshow(
        deltas,
        x=heatmap_metrics,
        y=scens,
        color_continuous_scale="RdYlGn_r",
        zmin=-vbound, zmax=vbound,
        text_auto=".1f",
        aspect="auto",
        labels={"color": "% Δ"},
        title=f"% delta vs {_BASELINE_NAME} (negative = better)",
    )
    wandb.log({"bench/delta_heatmap": fig})
    # Pin the exact ckpt used so the bench result is reproducible from the wandb run page.
    wandb.run.summary["bench/checkpoint"] = str(ckpt_path)
    wandb.finish()


def main(args: Args) -> None:
    manifest, scenarios = load_policy_dir(args.policy_dir)
    dt = 1.0 / manifest["policy_hz"]
    sim_dt = 1.0 / manifest["sim_hz"]

    print(f"# {args.policy_dir.name}")

    rows = [{"scenario": name, **compute_scenario_metrics(lo, hi, dt, sim_dt)}
            for name, lo, hi in scenarios]
    metric_names = [k for k in rows[0].keys() if k != "scenario"]

    # avg row averages only over scenarios where each metric is relevant,
    # so e.g. avg/vyaw_rmse is the mean over yaw_left/yaw_right only.
    avg = {"scenario": "avg"}
    for m in metric_names:
        vals = [r[m] for r in rows if _is_relevant(r["scenario"], m)]
        avg[m] = float(np.mean(vals)) if vals else float("nan")
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

    if args.resume_from_ckpt:
        _log_to_wandb(rows, metric_names, args.resume_from_ckpt, args.policy_dir)


if __name__ == "__main__":
    main(tyro.cli(Args))
