#!/usr/bin/env python3
"""CPU-only structural checks for the trajectory scenario generator.

This bypasses Env.__init__ and CUDA rendering. It checks reference geometry,
timing, trajectory features, scenario metadata, and obstacle insertion.

Run after applying the patch:
    python test_trajectory_scenarios.py --env-file env_cuda.py
"""
from __future__ import annotations

import argparse
import importlib.util
import math
from pathlib import Path
import sys
import types

import matplotlib.pyplot as plt
import numpy as np
import torch


def load_env_class(env_file: Path):
    # env_cuda.py imports the compiled extension at module import time. The
    # structural tests below do not call it, so a stub is sufficient.
    sys.modules.setdefault("quadsim_cuda", types.SimpleNamespace())
    spec = importlib.util.spec_from_file_location("env_cuda_tested", env_file)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {env_file}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.Env


def make_fake_env(Env, scenario: str, batch_size: int = 1):
    env = Env.__new__(Env)
    env.batch_size = batch_size
    env.device = torch.device("cpu")
    env.p = torch.zeros((batch_size, 3), dtype=torch.float32)
    env.p[:, 2] = 1.0
    env.p_target = torch.zeros((batch_size, 3), dtype=torch.float32)
    env.p_target[:, 0] = 8.0
    env.p_target[:, 2] = 1.0
    env.max_speed = torch.full((batch_size, 1), 3.0)
    env.voxels = torch.zeros((batch_size, 0, 6), dtype=torch.float32)

    env.random_rotation = False
    env.z_min = 0.4
    env.z_max = 4.0

    env.traj_scenario = scenario
    env.traj_scenario_probs = torch.ones(10, dtype=torch.float32) / 10
    env.traj_waypoints = 9
    env.traj_points = 8
    env.traj_dt = 0.25
    env.traj_pos_scale = 5.0
    env.traj_time_scale = 2.0
    return env


def wrapped_angle_delta(yaw):
    delta = yaw[..., -1] - yaw[..., 0]
    return torch.atan2(torch.sin(delta), torch.cos(delta))


def check_scenario(Env, scenario: str):
    torch.manual_seed(100 + Env.TRAJ_SCENARIOS.index(scenario))
    env = make_fake_env(Env, scenario)
    env._build_reference_trajectory()

    points = env.traj_waypoints_world[0]
    times = env.traj_times[0]
    yaw = env.traj_seg_yaw[0]

    assert points.shape == (env.traj_waypoints, 3)
    assert times.shape == (env.traj_waypoints,)
    assert torch.isfinite(points).all()
    assert torch.isfinite(times).all()
    assert torch.all(times[1:] > times[:-1])
    assert torch.allclose(env.p[0], points[0])
    assert torch.allclose(env.p_target[0], points[-1])

    R = torch.eye(3).unsqueeze(0)
    features = env.get_traj_features(0.0, R)
    velocity = env.get_reference_velocity(0.0)
    assert features.shape == (1, env.traj_points, 6)
    assert velocity.shape == (1, 3)
    assert torch.isfinite(features).all()
    assert torch.isfinite(velocity).all()

    xy_excursion = float(points[:, 1].abs().max())
    z_gain = float(points[:, 2].max() - points[0, 2])
    final_z_error = float((points[-1, 2] - points[0, 2]).abs())
    yaw_change = abs(float(wrapped_angle_delta(yaw[None])[0]))

    if scenario == "straight":
        assert xy_excursion < 1e-5
        assert yaw_change < 1e-4
    elif scenario == "smooth_turn":
        assert math.radians(20) < yaw_change < math.radians(55)
    elif scenario == "sharp_yaw_90":
        assert yaw_change > math.radians(70)
    elif scenario == "s_curve":
        assert float(points[:, 1].max()) > 0.3
        assert float(points[:, 1].min()) < -0.3
    elif scenario == "climb_over_wall":
        assert z_gain > 1.0
        assert env.voxels.shape[1] == 1
    elif scenario == "descend_after_obstacle":
        assert z_gain > 0.8
        assert final_z_error < 0.15
        assert env.voxels.shape[1] == 1
    elif scenario == "side_gap":
        assert 0.6 < xy_excursion < 1.6
        assert env.voxels.shape[1] == 2
    elif scenario == "edge_gap":
        assert xy_excursion > 1.3
        assert env.voxels.shape[1] == 2
    elif scenario == "around_obstacle":
        assert xy_excursion > 0.9
        assert env.voxels.shape[1] == 1
    elif scenario == "time_shifted_crossing":
        assert float(env.traj_start_delay[0]) >= 0.4
        assert torch.isfinite(env.traj_crossing_time[0])

    return {
        "scenario": scenario,
        "duration_s": float(times[-1]),
        "xy_excursion_m": xy_excursion,
        "z_gain_m": z_gain,
        "yaw_change_deg": math.degrees(yaw_change),
        "obstacles": int(env.voxels.shape[1]),
        "points": points.numpy(),
        "voxels": env.voxels[0].numpy(),
    }


def plot_xy(results, output_dir: Path):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    for ax, result in zip(axes.flat, results):
        points = result["points"]
        ax.plot(points[:, 0], points[:, 1], marker="o")
        ax.scatter(points[0, 0], points[0, 1], marker="o", label="start")
        ax.scatter(points[-1, 0], points[-1, 1], marker="x", label="goal")

        for voxel in result["voxels"]:
            if voxel[0] < -50:
                continue
            cx, cy, _, rx, ry, _ = voxel
            rect = plt.Rectangle(
                (cx - rx, cy - ry),
                2 * rx,
                2 * ry,
                fill=False,
                linewidth=1.5,
            )
            ax.add_patch(rect)

        ax.set_title(result["scenario"])
        ax.set_xlabel("x, m")
        ax.set_ylabel("y, m")
        ax.axis("equal")
        ax.grid(True)

    path = output_dir / "trajectory_scenarios_xy.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def plot_xz(results, output_dir: Path):
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), constrained_layout=True)
    for ax, result in zip(axes.flat, results):
        points = result["points"]
        ax.plot(points[:, 0], points[:, 2], marker="o")
        ax.scatter(points[0, 0], points[0, 2], marker="o")
        ax.scatter(points[-1, 0], points[-1, 2], marker="x")

        for voxel in result["voxels"]:
            if voxel[0] < -50:
                continue
            cx, _, cz, rx, _, rz = voxel
            rect = plt.Rectangle(
                (cx - rx, cz - rz),
                2 * rx,
                2 * rz,
                fill=False,
                linewidth=1.5,
            )
            ax.add_patch(rect)

        ax.set_title(result["scenario"])
        ax.set_xlabel("x, m")
        ax.set_ylabel("z, m")
        ax.grid(True)

    path = output_dir / "trajectory_scenarios_xz.png"
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return path


def check_mixed_distribution(Env):
    torch.manual_seed(7)
    env = make_fake_env(Env, "mixed", batch_size=1000)
    env.random_rotation = True
    env._build_reference_trajectory()
    counts = torch.bincount(env.traj_scenario_id, minlength=10)
    assert torch.all(counts > 0)
    assert torch.isfinite(env.traj_waypoints_world).all()
    assert torch.all(env.traj_times[:, 1:] > env.traj_times[:, :-1])
    return counts.tolist()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--env-file", type=Path, default=Path("env_cuda.py"))
    parser.add_argument("--output-dir", type=Path, default=Path("traj_scenario_check"))
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    Env = load_env_class(args.env_file)

    results = [
        check_scenario(Env, scenario)
        for scenario in Env.TRAJ_SCENARIOS
    ]
    counts = check_mixed_distribution(Env)

    xy_path = plot_xy(results, args.output_dir)
    xz_path = plot_xz(results, args.output_dir)

    header = (
        f"{'scenario':28s} {'duration':>9s} {'|y|max':>9s} "
        f"{'z gain':>9s} {'yaw deg':>9s} {'obs':>5s}"
    )
    print(header)
    print("-" * len(header))
    for item in results:
        print(
            f"{item['scenario']:28s} "
            f"{item['duration_s']:9.3f} "
            f"{item['xy_excursion_m']:9.3f} "
            f"{item['z_gain_m']:9.3f} "
            f"{item['yaw_change_deg']:9.1f} "
            f"{item['obstacles']:5d}"
        )

    print("\nMixed-scenario counts:", counts)
    print("Saved:", xy_path)
    print("Saved:", xz_path)
    print("All structural trajectory checks passed.")


if __name__ == "__main__":
    main()
