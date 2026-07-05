import argparse
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
    parser.add_argument("--out_dir", default="rollout_vis")
    parser.add_argument("--traj_idx", type=int, default=0)

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--timesteps", type=int, default=150)
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

    # Эти аргументы могут быть в configs/single_agent.args,
    # но для визуализации они не нужны. parse_known_args их проигнорирует.
    args, _ = parser.parse_known_args()
    return args


def save_xyz_plot(traj, out_path):
    t = np.arange(len(traj))

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
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_speed_plot(speed, out_path):
    t = np.arange(len(speed))

    plt.figure(figsize=(10, 5))
    plt.plot(t, speed)
    plt.xlabel("timestep")
    plt.ylabel("speed")
    plt.title("Speed history")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_xy_plot(traj, out_path):
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
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_3d_plot(traj, out_path):
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
    plt.savefig(out_path, dpi=160)
    plt.close()


def save_depth_gif(depth_frames, out_path):
    frames = []

    for d in depth_frames:
        # d: H x W, чем ближе объект, тем меньше depth.
        # Для визуализации ограничиваем диапазон.
        d = np.clip(d, 0.3, 24.0)
        img = 1.0 - (d - 0.3) / (24.0 - 0.3)
        img = np.clip(img * 255.0, 0, 255).astype(np.uint8)
        frames.append(img)

    imageio.mimsave(out_path, frames, fps=15)


@torch.no_grad()
def main():
    args = parse_args()
    os.makedirs(args.out_dir, exist_ok=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("device:", device)

    env = Env(
        args.batch_size,
        64,
        48,
        args.grad_decay,
        device,
        fov_x_half_tan=args.fov_x_half_tan,
        single=args.single,
        gate=args.gate,
        ground_voxels=args.ground_voxels,
        scaffold=args.scaffold,
        speed_mtp=args.speed_mtp,
        random_rotation=args.random_rotation,
        cam_angle=args.cam_angle,
    )

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

    B = args.batch_size
    traj_idx = min(args.traj_idx, B - 1)

    env.reset()
    model.reset()

    p_history = []
    v_history = []
    depth_frames = []

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

        depth_frames.append(depth[traj_idx].detach().cpu().numpy())

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

    p_history = torch.stack(p_history)
    v_history = torch.stack(v_history)

    traj = p_history[:, traj_idx].detach().cpu().numpy()
    vel = v_history[:, traj_idx].detach().cpu().numpy()
    speed = np.linalg.norm(vel, axis=-1)

    np.save(os.path.join(args.out_dir, "trajectory.npy"), traj)
    np.save(os.path.join(args.out_dir, "velocity.npy"), vel)

    save_xyz_plot(traj, os.path.join(args.out_dir, "position_xyz.png"))
    save_xy_plot(traj, os.path.join(args.out_dir, "trajectory_xy.png"))
    save_3d_plot(traj, os.path.join(args.out_dir, "trajectory_3d.png"))
    save_speed_plot(speed, os.path.join(args.out_dir, "speed.png"))
    save_depth_gif(depth_frames, os.path.join(args.out_dir, "depth.gif"))

    print("Saved visualization to:", args.out_dir)
    print("Final position:", traj[-1])
    print("Average speed:", float(speed.mean()))
    print("Max speed:", float(speed.max()))


if __name__ == "__main__":
    main()
