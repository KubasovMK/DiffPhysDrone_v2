import argparse
import inspect
import json
import math
import os
from random import normalvariate

import imageio.v2 as imageio
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from env_cuda import Env
from model import Model


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--resume", required=True)
    parser.add_argument("--scenario", choices=["over_wall", "edge_gap"], required=True)
    parser.add_argument("--out_dir", default="scenario_vis")

    parser.add_argument("--num_trials", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--timesteps", type=int, default=150)
    parser.add_argument("--save_top_k", type=int, default=3)

    parser.add_argument("--grad_decay", type=float, default=0.4)
    parser.add_argument("--speed_mtp", type=float, default=1.0)
    parser.add_argument("--fov_x_half_tan", type=float, default=0.53)
    parser.add_argument("--cam_angle", type=int, default=10)

    parser.add_argument("--single", default=False, action="store_true")
    parser.add_argument("--gate", default=False, action="store_true")
    parser.add_argument("--ground_voxels", default=False, action="store_true")
    parser.add_argument("--scaffold", default=False, action="store_true")
    parser.add_argument("--random_rotation", default=False, action="store_true")
    parser.add_argument("--yaw_drift", default=False, action="store_true")
    parser.add_argument("--no_odom", default=False, action="store_true")

    # Если ты добавлял эти параметры в Env.
    parser.add_argument("--random_z", default=False, action="store_true")
    parser.add_argument("--z_min", type=float, default=0.4)
    parser.add_argument("--z_max", type=float, default=4.0)
    parser.add_argument("--random_z_prob", type=float, default=0.3)

    # Пороговые параметры для scoring.
    parser.add_argument("--center_width_ratio", type=float, default=0.30)
    parser.add_argument("--edge_width_ratio", type=float, default=0.22)
    parser.add_argument("--near_depth", type=float, default=4.0)
    parser.add_argument("--far_depth", type=float, default=12.0)

    # Для over_wall: насколько заметный подъём считать полезным.
    parser.add_argument("--min_z_gain", type=float, default=0.4)

    # parse_known_args нужен, чтобы не падать на лишних аргументах из configs/single_agent.args.
    args, _ = parser.parse_known_args()
    return args


def build_env(args, device):
    signature = inspect.signature(Env.__init__)
    valid = set(signature.parameters.keys())

    kwargs = {
        "batch_size": args.batch_size,
        "width": 64,
        "height": 48,
        "grad_decay": args.grad_decay,
        "device": device,
        "fov_x_half_tan": args.fov_x_half_tan,
        "single": args.single,
        "gate": args.gate,
        "ground_voxels": args.ground_voxels,
        "scaffold": args.scaffold,
        "speed_mtp": args.speed_mtp,
        "random_rotation": args.random_rotation,
        "cam_angle": args.cam_angle,
        "random_z": args.random_z,
        "z_min": args.z_min,
        "z_max": args.z_max,
        "random_z_prob": args.random_z_prob,
    }

    filtered_kwargs = {k: v for k, v in kwargs.items() if k in valid}
    return Env(**filtered_kwargs)


def try_force_scenario(env, scenario):
    """
    Если в твоём env_cuda.py уже есть специальные методы генерации сценариев,
    этот блок попробует их вызвать.

    Если методов нет — ничего страшного, скрипт просто будет искать похожие случаи
    по depth-map и траектории.
    """
    if scenario == "over_wall":
        candidate_names = [
            "_add_over_wall_scenario",
            "add_over_wall_scenario",
            "_make_over_wall_scenario",
            "make_over_wall_scenario",
        ]
    else:
        candidate_names = [
            "_add_edge_gap_scenario",
            "add_edge_gap_scenario",
            "_add_edge_free_space_scenario",
            "add_edge_free_space_scenario",
            "_add_free_space_edge_scenario",
            "add_free_space_edge_scenario",
        ]

    for name in candidate_names:
        if hasattr(env, name):
            fn = getattr(env, name)
            print(f"[scenario hook] trying env.{name}()")

            for call_variant in range(3):
                try:
                    if call_variant == 0:
                        fn()
                    elif call_variant == 1:
                        idx = torch.arange(env.p.shape[0], device=env.p.device)
                        fn(idx)
                    else:
                        fn(indices=None)

                    print(f"[scenario hook] applied: {name}")
                    return True

                except TypeError:
                    continue
                except Exception as e:
                    print(f"[scenario hook] {name} failed: {repr(e)}")
                    return False

    print("[scenario hook] no matching scenario method found; using heuristic search")
    return False


def normalize_depth_for_gif(depth):
    depth = np.clip(depth, 0.3, 24.0)
    img = 1.0 - (depth - 0.3) / (24.0 - 0.3)
    img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    return img


def save_depth_gif(depth_frames, out_path):
    frames = [normalize_depth_for_gif(d) for d in depth_frames]
    imageio.mimsave(out_path, frames, fps=15)


def save_depth_contact_sheet(depth_frames, out_path, n=12):
    if len(depth_frames) == 0:
        return

    idxs = np.linspace(0, len(depth_frames) - 1, min(n, len(depth_frames))).astype(int)

    cols = 4
    rows = int(math.ceil(len(idxs) / cols))

    plt.figure(figsize=(4 * cols, 3 * rows))

    for plot_i, frame_i in enumerate(idxs):
        plt.subplot(rows, cols, plot_i + 1)
        plt.imshow(depth_frames[frame_i], cmap="gray_r")
        plt.title(f"t={frame_i}")
        plt.axis("off")

    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_trajectory_plots(traj, vel, out_dir):
    t = np.arange(len(traj))
    speed = np.linalg.norm(vel, axis=-1)

    plt.figure(figsize=(10, 5))
    plt.plot(t, traj[:, 0], label="x")
    plt.plot(t, traj[:, 1], label="y")
    plt.plot(t, traj[:, 2], label="z")
    plt.xlabel("timestep")
    plt.ylabel("position")
    plt.title("Position history")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "position_xyz.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(10, 5))
    plt.plot(t, speed)
    plt.xlabel("timestep")
    plt.ylabel("speed")
    plt.title("Speed history")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "speed.png"), dpi=160)
    plt.close()

    plt.figure(figsize=(7, 7))
    plt.plot(traj[:, 0], traj[:, 1], marker="o", markersize=2)
    plt.scatter(traj[0, 0], traj[0, 1], marker="o", s=80, label="start")
    plt.scatter(traj[-1, 0], traj[-1, 1], marker="x", s=100, label="finish")
    plt.xlabel("x")
    plt.ylabel("y")
    plt.title("Top-down trajectory: x-y")
    plt.axis("equal")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory_xy.png"), dpi=160)
    plt.close()

    fig = plt.figure(figsize=(8, 7))
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(traj[:, 0], traj[:, 1], traj[:, 2], marker="o", markersize=2)
    ax.scatter(traj[0, 0], traj[0, 1], traj[0, 2], s=80, label="start")
    ax.scatter(traj[-1, 0], traj[-1, 1], traj[-1, 2], s=100, marker="x", label="finish")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("3D trajectory")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, "trajectory_3d.png"), dpi=160)
    plt.close()


def compute_depth_regions(depth_seq, center_width_ratio=0.30, edge_width_ratio=0.22):
    """
    depth_seq: T x H x W
    Возвращает средние depth-значения в центре, слева и справа.
    """
    T, H, W = depth_seq.shape

    center_w = max(1, int(W * center_width_ratio))
    edge_w = max(1, int(W * edge_width_ratio))

    c0 = W // 2 - center_w // 2
    c1 = W // 2 + center_w // 2

    left = depth_seq[:, :, :edge_w]
    right = depth_seq[:, :, W - edge_w:]
    center = depth_seq[:, :, c0:c1]

    left_mean = left.mean(axis=(1, 2))
    right_mean = right.mean(axis=(1, 2))
    center_mean = center.mean(axis=(1, 2))

    return left_mean, center_mean, right_mean


def score_edge_gap(depth_seq, traj, args):
    """
    Высокий score, если:
    - центр depth-map ближе/закрыт;
    - левый или правый край более свободен;
    - дрон смещается вбок.
    """
    left_mean, center_mean, right_mean = compute_depth_regions(
        depth_seq,
        center_width_ratio=args.center_width_ratio,
        edge_width_ratio=args.edge_width_ratio,
    )

    best_edge = np.maximum(left_mean, right_mean)
    edge_advantage = best_edge - center_mean

    # Берём максимум по времени: нас интересует, был ли хотя бы один явно edge-gap момент.
    max_edge_advantage = float(edge_advantage.max())

    # Центр должен быть относительно близким, иначе это просто пустая сцена.
    center_near_score = float(np.maximum(0.0, args.far_depth - center_mean).max())

    # Боковой манёвр.
    lateral_shift = float(np.abs(traj[-1, 1] - traj[0, 1]))
    max_lateral_excursion = float(np.max(np.abs(traj[:, 1] - traj[0, 1])))

    score = (
        2.0 * max_edge_advantage
        + 0.25 * center_near_score
        + 0.5 * lateral_shift
        + 0.5 * max_lateral_excursion
    )

    return score, {
        "max_edge_advantage": max_edge_advantage,
        "center_near_score": center_near_score,
        "lateral_shift": lateral_shift,
        "max_lateral_excursion": max_lateral_excursion,
        "left_mean_min": float(left_mean.min()),
        "left_mean_max": float(left_mean.max()),
        "center_mean_min": float(center_mean.min()),
        "center_mean_max": float(center_mean.max()),
        "right_mean_min": float(right_mean.min()),
        "right_mean_max": float(right_mean.max()),
    }


def score_over_wall(depth_seq, traj, vel, args):
    """
    Высокий score, если:
    - в начале/середине rollout в центре depth-map есть близкое препятствие;
    - дрон заметно увеличивает z;
    - дрон продолжает двигаться вперёд.
    """
    _, center_mean, _ = compute_depth_regions(
        depth_seq,
        center_width_ratio=args.center_width_ratio,
        edge_width_ratio=args.edge_width_ratio,
    )

    z = traj[:, 2]
    x = traj[:, 0]
    speed = np.linalg.norm(vel, axis=-1)

    z_gain = float(z.max() - z[0])
    final_z_delta = float(z[-1] - z[0])
    forward_progress = float(x[-1] - x[0])
    avg_speed = float(speed.mean())

    # Чем меньше center_mean, тем ближе препятствие.
    center_obstacle_score = float(np.maximum(0.0, args.near_depth - center_mean).max())

    # Награда только если есть заметный подъём.
    z_gain_score = max(0.0, z_gain - args.min_z_gain)

    score = (
        3.0 * z_gain_score
        + 1.0 * center_obstacle_score
        + 0.25 * forward_progress
        + 0.1 * avg_speed
    )

    return score, {
        "z_gain": z_gain,
        "final_z_delta": final_z_delta,
        "forward_progress": forward_progress,
        "avg_speed": avg_speed,
        "max_speed": float(speed.max()),
        "center_obstacle_score": center_obstacle_score,
        "center_depth_min": float(center_mean.min()),
        "center_depth_max": float(center_mean.max()),
    }


@torch.no_grad()
def run_one_trial(args, model, env, device, trial_id):
    B = args.batch_size

    env.reset()
    model.reset()

    # Если в env_cuda.py есть специальный генератор сценария, пробуем применить.
    try_force_scenario(env, args.scenario)

    p_history = []
    v_history = []
    depth_history = []

    h = None
    act_lag = 1
    act_buffer = [env.act] * (act_lag + 1)

    target_v_raw = env.p_target - env.p

    if args.yaw_drift:
        drift_av = torch.randn(B, device=device) * (5 * math.pi / 180 / 15)
        zeros = torch.zeros_like(drift_av)
        ones = torch.ones_like(drift_av)
        R_drift = torch.stack(
            [
                torch.cos(drift_av), -torch.sin(drift_av), zeros,
                torch.sin(drift_av), torch.cos(drift_av), zeros,
                zeros, zeros, ones,
            ],
            -1,
        ).reshape(B, 3, 3)
    else:
        R_drift = None

    for t in range(args.timesteps):
        ctl_dt = normalvariate(1 / 15, 0.1 / 15)

        depth, flow = env.render(ctl_dt)

        p_history.append(env.p.detach().clone())
        v_history.append(env.v.detach().clone())
        depth_history.append(depth.detach().clone())

        if args.yaw_drift:
            target_v_raw = torch.squeeze(target_v_raw[:, None] @ R_drift, 1)
        else:
            target_v_raw = env.p_target - env.p.detach()

        env.run(act_buffer[t], ctl_dt, target_v_raw)

        R = env.R
        fwd = env.R[:, :, 0].clone()
        up = torch.zeros_like(fwd)

        fwd[:, 2] = 0
        up[:, 2] = 1

        fwd = F.normalize(fwd, 2, -1)
        R = torch.stack([fwd, torch.cross(up, fwd, dim=-1), up], -1)

        target_v_norm = torch.norm(target_v_raw, 2, -1, keepdim=True)
        target_v_unit = target_v_raw / target_v_norm.clamp_min(1e-6)
        target_v = target_v_unit * torch.minimum(target_v_norm, env.max_speed)

        state = [
            torch.squeeze(target_v[:, None] @ R, 1),
            env.R[:, 2],
            env.margin[:, None],
        ]

        local_v = torch.squeeze(env.v[:, None] @ R, 1)

        if not args.no_odom:
            state.insert(0, local_v)

        state = torch.cat(state, -1)

        x = 3 / depth.clamp(0.3, 24) - 0.6
        x = F.max_pool2d(x[:, None], 4, 4)

        act, _, h = model(x, state, h)

        a_pred, v_pred, *_ = (R @ act.reshape(B, 3, -1)).unbind(-1)
        act = (a_pred - v_pred - env.g_std) * env.thr_est_error[:, None] + env.g_std

        act_buffer.append(act)

    p_history = torch.stack(p_history).detach().cpu().numpy()       # T x B x 3
    v_history = torch.stack(v_history).detach().cpu().numpy()       # T x B x 3
    depth_history = torch.stack(depth_history).detach().cpu().numpy()  # T x B x H x W

    candidates = []

    for b in range(B):
        traj = p_history[:, b, :]
        vel = v_history[:, b, :]
        depth_seq = depth_history[:, b, :, :]

        if args.scenario == "edge_gap":
            score, metrics = score_edge_gap(depth_seq, traj, args)
        else:
            score, metrics = score_over_wall(depth_seq, traj, vel, args)

        candidates.append({
            "trial_id": trial_id,
            "batch_idx": b,
            "score": float(score),
            "metrics": metrics,
            "traj": traj,
            "vel": vel,
            "depth_seq": depth_seq,
        })

    return candidates


def save_case(case, case_dir):
    os.makedirs(case_dir, exist_ok=True)

    traj = case["traj"]
    vel = case["vel"]
    depth_seq = case["depth_seq"]

    np.save(os.path.join(case_dir, "trajectory.npy"), traj)
    np.save(os.path.join(case_dir, "velocity.npy"), vel)
    np.save(os.path.join(case_dir, "depth.npy"), depth_seq)

    save_trajectory_plots(traj, vel, case_dir)
    save_depth_gif(depth_seq, os.path.join(case_dir, "depth.gif"))
    save_depth_contact_sheet(depth_seq, os.path.join(case_dir, "depth_contact_sheet.png"))

    meta = {
        "trial_id": case["trial_id"],
        "batch_idx": case["batch_idx"],
        "score": case["score"],
        "metrics": case["metrics"],
    }

    with open(os.path.join(case_dir, "metrics.json"), "w") as f:
        json.dump(meta, f, indent=2)


def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    env = build_env(args, device)

    if args.no_odom:
        model = Model(7, 6)
    else:
        model = Model(7 + 3, 6)

    model = model.to(device)

    state_dict = torch.load(args.resume, map_location=device)
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)

    if missing_keys:
        print("missing_keys:", missing_keys)
    if unexpected_keys:
        print("unexpected_keys:", unexpected_keys)

    model.eval()

    all_cases = []

    for trial_id in range(args.num_trials):
        print(f"trial {trial_id + 1}/{args.num_trials}")
        cases = run_one_trial(args, model, env, device, trial_id)
        all_cases.extend(cases)

        # Сохраняем промежуточный топ, чтобы не потерять результат при остановке.
        all_cases_sorted = sorted(all_cases, key=lambda x: x["score"], reverse=True)
        top_cases = all_cases_sorted[:args.save_top_k]

        summary = []
        for rank, case in enumerate(top_cases):
            summary.append({
                "rank": rank,
                "trial_id": case["trial_id"],
                "batch_idx": case["batch_idx"],
                "score": case["score"],
                "metrics": case["metrics"],
            })

        with open(os.path.join(args.out_dir, "summary_live.json"), "w") as f:
            json.dump(summary, f, indent=2)

    all_cases = sorted(all_cases, key=lambda x: x["score"], reverse=True)
    top_cases = all_cases[:args.save_top_k]

    final_summary = []

    for rank, case in enumerate(top_cases):
        case_dir = os.path.join(args.out_dir, f"case_{rank:02d}")
        save_case(case, case_dir)

        final_summary.append({
            "rank": rank,
            "trial_id": case["trial_id"],
            "batch_idx": case["batch_idx"],
            "score": case["score"],
            "metrics": case["metrics"],
            "case_dir": case_dir,
        })

        print(f"[saved] rank={rank} score={case['score']:.3f} dir={case_dir}")

    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(final_summary, f, indent=2)

    print("Done.")
    print("Output directory:", args.out_dir)


if __name__ == "__main__":
    main()
