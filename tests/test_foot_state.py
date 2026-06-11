from __future__ import annotations

import math

import pytest
import torch

from holosoma.utils.foot_state import FootState

DT = 0.02


def _make_state() -> FootState:
    return FootState(num_envs=1, num_feet=2, dt=DT, device="cpu")


def _step(state: FootState, vz_left: float, fz_left: float) -> dict:
    vz = torch.zeros(1, 2)
    fz = torch.zeros(1, 2)
    vz[0, 0] = vz_left
    fz[0, 0] = fz_left
    state.step(vz, fz)
    log: dict = {}
    state.log(log)
    return log


def test_touchdown_fires_and_captures_pre_fire_velocity() -> None:
    state = _make_state()

    log = _step(state, vz_left=-0.8, fz_left=0.0)
    assert not state.fired.any()

    log = _step(state, vz_left=-0.1, fz_left=2.0)
    assert state.fired[0, 0]
    assert state.vz_pre[0, 0].item() == pytest.approx(-0.8)
    assert state.approach_speed[0, 0].item() == pytest.approx(0.8)
    assert log["feet/touchdown_rate"].item() == 1.0
    assert log["feet/touchdown_foot_rate"].item() == 0.5
    assert log["feet/vz_at_pre"].item() == pytest.approx(-0.8)
    assert log["feet/approach_speed"].item() == pytest.approx(0.8)
    assert log["feet/fz_at_fire"].item() == 2.0
    assert log["feet/contact_rate"].item() == 0.5
    assert log["feet/contact_duration_s"].item() == pytest.approx(DT)
    assert log["feet/left_vz_at_pre"].item() == pytest.approx(-0.8)
    assert math.isnan(log["feet/right_vz_at_pre"].item())


def test_reset_clears_detector_and_pre_fire_buffer() -> None:
    state = _make_state()

    _step(state, vz_left=-0.8, fz_left=0.0)
    state.reset()
    log = _step(state, vz_left=-0.1, fz_left=2.0)

    assert not state.fired.any()
    assert state.vz_pre[0, 0].item() == 0.0
    assert log["feet/touchdown_rate"].item() == 0.0
    assert math.isnan(log["feet/vz_at_pre"].item())


def test_reset_with_env_ids_clears_only_those_envs() -> None:
    state = FootState(num_envs=2, num_feet=2, dt=DT, device="cpu")
    vz = torch.full((2, 2), -0.8)
    fz = torch.zeros(2, 2)
    state.step(vz, fz)

    state.reset(torch.tensor([0]))
    state.step(torch.full((2, 2), -0.1), torch.full((2, 2), 2.0))

    assert not state.fired[0].any()  # env 0 disarmed by reset
    assert state.fired[1].all()
    assert state.vz_pre[0, 0].item() == 0.0
    assert state.vz_pre[1, 0].item() == pytest.approx(-0.8)


def test_no_event_logs_zero_rates() -> None:
    state = _make_state()

    log = _step(state, vz_left=-0.1, fz_left=0.0)

    assert log["feet/touchdown_rate"].item() == 0.0
    assert log["feet/contact_rate"].item() == 0.0
    assert log["feet/active_force_window_rate"].item() == 0.0


def test_force_window_logs_deferral_observability() -> None:
    state = _make_state()

    _step(state, vz_left=-0.8, fz_left=0.0)
    _step(state, vz_left=-0.1, fz_left=2.0)
    _step(state, vz_left=-0.1, fz_left=10.0)
    log = _step(state, vz_left=-0.1, fz_left=8.0)

    assert log["feet/fz_peak_60ms"].item() == 10.0
    assert log["feet/impulse_60ms"].item() == pytest.approx(0.4)
    assert log["feet/force_deferral_ratio_60ms"].item() == 5.0
