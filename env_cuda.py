import math
import random
import time
import torch
import torch.nn.functional as F
import quadsim_cuda


class GDecay(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, alpha):
        ctx.alpha = alpha
        return x

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output * ctx.alpha, None

g_decay = GDecay.apply


class RunFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, grad_decay, ctl_dt, airmode):
        act_next, p_next, v_next, a_next = quadsim_cuda.run_forward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, act_pred, act, p, v, v_wind, a, ctl_dt, airmode)
        ctx.save_for_backward(R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next)
        ctx.grad_decay = grad_decay
        ctx.ctl_dt = ctl_dt
        return act_next, p_next, v_next, a_next

    @staticmethod
    def backward(ctx, d_act_next, d_p_next, d_v_next, d_a_next):
        R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next = ctx.saved_tensors
        d_act_pred, d_act, d_p, d_v, d_a = quadsim_cuda.run_backward(
            R, dg, z_drag_coef, drag_2, pitch_ctl_delay, v, v_wind, act_next, d_act_next, d_p_next, d_v_next, d_a_next,
            ctx.grad_decay, ctx.ctl_dt)
        return None, None, None, None, None, d_act_pred, d_act, d_p, d_v, None, d_a, None, None, None

run = RunFunction.apply


class Env:
    def __init__(self, batch_size, width, height, grad_decay, device='cpu', fov_x_half_tan=0.53,
                 single=False, gate=False, ground_voxels=False, scaffold=False, speed_mtp=1,
                 random_rotation=False, cam_angle=10, random_z=False, z_min = 0.4, z_max = 4.0, 
                 random_z_prob = 0.3, over_wall=False, edge_gap=False, over_wall_prob=0.0, edge_gap_prob=0.0, 
                 edge_gap_block_ratio_min=0.85, edge_gap_block_ratio_max=0.90, edge_gap_aim_target=True,
                 traj_points=8, traj_dt=0.25, traj_pos_scale=5.0, traj_time_scale=2.0,) -> None:     #added random_z, z_min, z_max
        self.device = device
        self.batch_size = batch_size
        self.width = width
        self.height = height
        self.grad_decay = grad_decay
        self.ball_w = torch.tensor([8., 18, 6, 0.2], device=device)
        self.ball_b = torch.tensor([0., -9, -1, 0.4], device=device)
        self.voxel_w = torch.tensor([8., 18, 6, 0.1, 0.1, 0.1], device=device)
        self.voxel_b = torch.tensor([0., -9, -1, 0.2, 0.2, 0.2], device=device)
        self.ground_voxel_w = torch.tensor([8., 18,  0, 2.9, 2.9, 1.9], device=device)
        self.ground_voxel_b = torch.tensor([0., -9, -1, 0.1, 0.1, 0.1], device=device)
        self.cyl_w = torch.tensor([8., 18, 0.35], device=device)
        self.cyl_b = torch.tensor([0., -9, 0.05], device=device)
        self.cyl_h_w = torch.tensor([8., 6, 0.1], device=device)
        self.cyl_h_b = torch.tensor([0., 0, 0.05], device=device)
        self.gate_w = torch.tensor([2.,  2,  1.0, 0.5], device=device)
        self.gate_b = torch.tensor([3., -1,  0.0, 0.5], device=device)
        self.v_wind_w = torch.tensor([1,  1,  0.2], device=device)
        self.g_std = torch.tensor([0., 0, -9.80665], device=device)
        self.roof_add = torch.tensor([0., 0., 2.5, 1.5, 1.5, 1.5], device=device)
        self.sub_div = torch.linspace(0, 1. / 15, 10, device=device).reshape(-1, 1, 1)
        self.p_init = torch.as_tensor([
            [-1.5, -3.,  1],
            [ 9.5, -3.,  1],
            [-0.5,  1.,  1],
            [ 8.5,  1.,  1],
            [ 0.0,  3.,  1],
            [ 8.0,  3.,  1],
            [-1.0, -1.,  1],
            [ 9.0, -1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        self.p_end = torch.as_tensor([
            [8.,  3.,  1],
            [0.,  3.,  1],
            [8., -1.,  1],
            [0., -1.,  1],
            [8., -3.,  1],
            [0., -3.,  1],
            [8.,  1.,  1],
            [0.,  1.,  1],
        ], device=device).repeat(batch_size // 8 + 7, 1)[:batch_size]
        self.flow = torch.empty((batch_size, 0, height, width), device=device)
        self.single = single
        self.gate = gate
        self.ground_voxels = ground_voxels
        self.scaffold = scaffold
        self.speed_mtp = speed_mtp
        self.random_rotation = random_rotation
        self.cam_angle = cam_angle
        self.fov_x_half_tan = fov_x_half_tan

        #z-axis randomization  
        self.random_z = random_z
        self.z_min = z_min
        self.z_max = z_max
        self.random_z_prob = random_z_prob

        #custom obstacle avoidance
        self.over_wall = over_wall
        self.edge_gap = edge_gap
        self.over_wall_prob = over_wall_prob
        self.edge_gap_prob = edge_gap_prob

        self.edge_gap_block_ratio_min = edge_gap_block_ratio_min
        self.edge_gap_block_ratio_max = edge_gap_block_ratio_max
        self.edge_gap_aim_target = edge_gap_aim_target

        #trajectory segments
        self.traj_points = traj_points
        self.traj_dt = traj_dt
        self.traj_pos_scale = traj_pos_scale
        self.traj_time_scale = traj_time_scale
        
        self.reset()
        # self.obj_avoid_grad_mtp = torch.tensor([0.5, 2., 1.], device=device)

    def reset(self):
        B = self.batch_size
        device = self.device

        cam_angle = (self.cam_angle + torch.randn(B, device=device)) * math.pi / 180
        zeros = torch.zeros_like(cam_angle)
        ones = torch.ones_like(cam_angle)
        self.R_cam = torch.stack([
            torch.cos(cam_angle), zeros, -torch.sin(cam_angle),
            zeros, ones, zeros,
            torch.sin(cam_angle), zeros, torch.cos(cam_angle),
        ], -1).reshape(B, 3, 3)

        # env
        self.balls = torch.rand((B, 30, 4), device=device) * self.ball_w + self.ball_b
        self.voxels = torch.rand((B, 30, 6), device=device) * self.voxel_w + self.voxel_b
        self.cyl = torch.rand((B, 30, 3), device=device) * self.cyl_w + self.cyl_b
        self.cyl_h = torch.rand((B, 2, 3), device=device) * self.cyl_h_w + self.cyl_h_b

        self._fov_x_half_tan = (0.95 + 0.1 * random.random()) * self.fov_x_half_tan
        self.n_drones_per_group = random.choice([4, 8])
        self.drone_radius = random.uniform(0.1, 0.15)
        if self.single:
            self.n_drones_per_group = 1

        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        self.max_speed = (0.75 + 2.5 * rd) * self.speed_mtp
        scale = (self.max_speed - 0.5).clamp_min(1)

        self.thr_est_error = 1 + torch.randn(B, device=device) * 0.01

        roof = torch.rand((B,)) < 0.5
        self.balls[~roof, :15, :2] = self.cyl[~roof, :15, :2]
        self.voxels[~roof, :15, :2] = self.cyl[~roof, 15:, :2]
        self.balls[~roof, :15] = self.balls[~roof, :15] + self.roof_add[:4]
        self.voxels[~roof, :15] = self.voxels[~roof, :15] + self.roof_add
        self.balls[..., 0] = torch.minimum(torch.maximum(self.balls[..., 0], self.balls[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.balls[..., 3])
        self.voxels[..., 0] = torch.minimum(torch.maximum(self.voxels[..., 0], self.voxels[..., 3] + 0.3 / scale), 8 - 0.3 / scale - self.voxels[..., 3])
        self.cyl[..., 0] = torch.minimum(torch.maximum(self.cyl[..., 0], self.cyl[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl[..., 2])
        self.cyl_h[..., 0] = torch.minimum(torch.maximum(self.cyl_h[..., 0], self.cyl_h[..., 2] + 0.3 / scale), 8 - 0.3 / scale - self.cyl_h[..., 2])
        self.voxels[roof, 0, 2] = self.voxels[roof, 0, 2] * 0.5 + 201
        self.voxels[roof, 0, 3:] = 200

        if self.ground_voxels:
            ground_balls_r = 8 + torch.rand((B, 2), device=device) * 6
            ground_balls_r_ground = 2 + torch.rand((B, 2), device=device) * 4
            ground_balls_h = ground_balls_r - (ground_balls_r.pow(2) - ground_balls_r_ground.pow(2)).sqrt()
            # |   ground_balls_h
            # ----- ground_balls_r_ground
            # |  /
            # | / ground_balls_r
            # |/
            self.balls[:, :2, 3] = ground_balls_r
            self.balls[:, :2, 2] = ground_balls_h - ground_balls_r - 1

            # planner shape in (0.1-2.0) times (0.1-2.0)
            ground_voxels = torch.rand((B, 10, 6), device=device) * self.ground_voxel_w + self.ground_voxel_b
            ground_voxels[:, :, 2] = ground_voxels[:, :, 5] - 1
            self.voxels = torch.cat([self.voxels, ground_voxels], 1)

        self.voxels[:, :, 1] *= (self.max_speed + 4) / scale
        self.balls[:, :, 1] *= (self.max_speed + 4) / scale
        self.cyl[:, :, 1] *= (self.max_speed + 4) / scale

        # gates
        if self.gate:
            gate = torch.rand((B, 4), device=device) * self.gate_w + self.gate_b
            p = gate[None, :, :3]
            nearest_pt = torch.empty_like(p)
            quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, 1)
            gate_x, gate_y, gate_z, gate_r = gate.unbind(-1)
            gate_x[(nearest_pt - p).norm(2, -1)[0] < 0.5] = -50
            ones = torch.ones_like(gate_x)
            gate = torch.stack([
                torch.stack([gate_x, gate_y + gate_r + 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z + gate_r + 5, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y - gate_r - 5, gate_z, ones * 0.05, ones * 5, ones * 5], -1),
                torch.stack([gate_x, gate_y, gate_z - gate_r - 5, ones * 0.05, ones * 5, ones * 5], -1),
            ], 1)

            self.voxels = torch.cat([self.voxels, gate], 1)
        self.voxels[..., 0] *= scale
        self.balls[..., 0] *= scale
        self.cyl[..., 0] *= scale
        self.cyl_h[..., 0] *= scale
        if self.ground_voxels:
            self.balls[:, :2, 0] = torch.minimum(torch.maximum(self.balls[:, :2, 0], ground_balls_r_ground + 0.3), scale * 8 - 0.3 - ground_balls_r_ground)

        # drone
        self.pitch_ctl_delay = 12 + 1.2 * torch.randn((B, 1), device=device)
        self.yaw_ctl_delay = 6 + 0.6 * torch.randn((B, 1), device=device)

        rd = torch.rand((B // self.n_drones_per_group, 1), device=device).repeat_interleave(self.n_drones_per_group, 0)
        scale = torch.cat([
            scale,
            rd + 0.5,
            torch.rand_like(scale) - 0.5], -1)
        self.p = self.p_init * scale + torch.randn_like(scale) * 0.1
        self.p_target = self.p_end * scale + torch.randn_like(scale) * 0.1

        #added wall across trajectory
        wall_prob = 0.3
        mask = torch.rand(B, device=device) < wall_prob


        if self.random_z:   #added randomization on z-axis not for 100% batch
            mask = torch.rand((B, ), device=device) < self.random_z_prob

            z_init = torch.rand((B,), device=device) * (self.z_max - self.z_min) + self.z_min
            z_target = torch.rand((B,), device=device) * (self.z_max - self.z_min) + self.z_min

            self.p[mask, 2] = z_init[mask]
            self.p_target[mask, 2] = z_target[mask]

        if self.random_rotation:
            yaw_bias = torch.rand(B//self.n_drones_per_group, device=device).repeat_interleave(self.n_drones_per_group, 0) * 1.5 - 0.75
            c = torch.cos(yaw_bias)
            s = torch.sin(yaw_bias)
            l = torch.ones_like(yaw_bias)
            o = torch.zeros_like(yaw_bias)
            R = torch.stack([c,-s, o, s, c, o, o, o, l], -1).reshape(B, 3, 3)
            self.p = torch.squeeze(R @ self.p[..., None], -1)
            self.p_target = torch.squeeze(R @ self.p_target[..., None], -1)
            self.voxels[..., :3] = (R @ self.voxels[..., :3].transpose(1, 2)).transpose(1, 2)
            self.balls[..., :3] = (R @ self.balls[..., :3].transpose(1, 2)).transpose(1, 2)
            self.cyl[..., :3] = (R @ self.cyl[..., :3].transpose(1, 2)).transpose(1, 2)

        #custom obstacles
        if self.over_wall:
            self._add_over_wall_scenario(prob=self.over_wall_prob)

        if self.edge_gap:
            self._add_edge_gap_scenario(prob=self.edge_gap_prob)

        #reference trajectory
        self._build_reference_trajectory()

        # scaffold
        if self.scaffold and random.random() < 0.5:
            x = torch.arange(1, 6, dtype=torch.float, device=device)
            y = torch.arange(-3, 4, dtype=torch.float, device=device)
            z = torch.arange(1, 4, dtype=torch.float, device=device)
            _x, _y = torch.meshgrid(x, y)
            # + torch.rand_like(self.max_speed) * self.max_speed
            # + torch.randn_like(self.max_speed)
            scaf_v = torch.stack([_x, _y, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            x_bias = torch.rand_like(self.max_speed) * self.max_speed
            scale = 1 + torch.rand((B, 1, 1), device=device)
            scaf_v = scaf_v * scale + torch.stack([
                x_bias,
                torch.randn_like(self.max_speed),
                torch.rand_like(self.max_speed) * 0.01
            ], -1)
            self.cyl = torch.cat([self.cyl, scaf_v], 1)
            _x, _z = torch.meshgrid(x, z)
            scaf_h = torch.stack([_x, _z, torch.full_like(_x, 0.02)], -1).flatten(0, 1)
            scaf_h = scaf_h * scale + torch.stack([
                x_bias,
                torch.randn_like(self.max_speed) * 0.1,
                torch.rand_like(self.max_speed) * 0.01
            ], -1)
            self.cyl_h = torch.cat([self.cyl_h, scaf_h], 1)

        self.v = torch.randn((B, 3), device=device) * 0.2
        self.v_wind = torch.randn((B, 3), device=device) * self.v_wind_w
        self.act = torch.randn_like(self.v) * 0.1
        self.a = self.act
        self.dg = torch.randn((B, 3), device=device) * 0.2

        R = torch.zeros((B, 3, 3), device=device)
        self.R = quadsim_cuda.update_state_vec(R, self.act, torch.randn((B, 3), device=device) * 0.2 + F.normalize(self.p_target - self.p),
            torch.zeros_like(self.yaw_ctl_delay), 5)
        self.R_old = self.R.clone()
        self.p_old = self.p
        self.margin = torch.rand((B,), device=device) * 0.2 + 0.1

        # drag coef
        self.drag_2 = torch.rand((B, 2), device=device) * 0.15 + 0.3
        self.drag_2[:, 0] = 0
        self.z_drag_coef = torch.ones((B, 1), device=device)

    @torch.no_grad()
    def _build_reference_trajectory(self):
        """
        Stores a simple time-parameterized reference trajectory for each sample.

        MVP version:
        straight line from current start position to current target position.
        Later this can be replaced by local planner trajectory:
        [(t, x, y, z, theta), ...].
        """
        self.traj_start = self.p.detach().clone()
        self.traj_goal = self.p_target.detach().clone()

        delta = self.traj_goal - self.traj_start
        dist = torch.norm(delta, dim=-1).clamp_min(0.1)

        max_speed = self.max_speed.reshape(-1).clamp_min(0.5)
        self.traj_total_time = (dist / max_speed).clamp_min(1.0)

        # Reference yaw is the direction of the path in XY.
        self.traj_yaw = torch.atan2(delta[:, 1], delta[:, 0])


    @torch.no_grad()
    def get_traj_features(self, sim_time, R_local):
        """
        Returns trajectory features with shape [B, K, 6].

        Feature per future point:
        [dt, dx_body, dy_body, dz_body, sin(dtheta), cos(dtheta)]

        R_local is the same local frame matrix used in main_cuda.py for state.
        """
        B = self.batch_size
        device = self.device
        dtype = self.p.dtype
        K = self.traj_points

        step_ids = torch.arange(1, K + 1, device=device, dtype=dtype)
        dt = step_ids[None, :] * self.traj_dt  # [1, K]

        future_time = float(sim_time) + dt  # [1, K]
        alpha = future_time / self.traj_total_time[:, None]
        alpha = alpha.clamp(0.0, 1.0)

        p_ref = self.traj_start[:, None, :] + alpha[:, :, None] * (
            self.traj_goal[:, None, :] - self.traj_start[:, None, :]
        )

        # Current position -> future reference points.
        dp_world = p_ref - self.p.detach()[:, None, :]

        # Convert world-frame displacement to local/body-like frame.
        # This matches the convention used in main_cuda.py:
        # local_vector = world_vector @ R
        dp_local = torch.bmm(dp_world, R_local)

        fwd = R_local[:, :, 0]
        yaw_now = torch.atan2(fwd[:, 1], fwd[:, 0])

        dtheta = self.traj_yaw[:, None] - yaw_now[:, None]
        dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))

        traj = torch.stack(
            [
                dt.expand(B, K) / self.traj_time_scale,
                dp_local[:, :, 0] / self.traj_pos_scale,
                dp_local[:, :, 1] / self.traj_pos_scale,
                dp_local[:, :, 2] / self.traj_pos_scale,
                torch.sin(dtheta),
                torch.cos(dtheta),
            ],
            dim=-1,
        )

        return traj

    @torch.no_grad()
    def _add_over_wall_scenario(self, prob = 0.25):
        '''
        Adds a wall that blocks direct path to target point.
        The intended behaviour is to fly over the wall

        self.voxels format: [cx, cy, cz, rx, ry, rz]
        c = box center, r = box half-size
        '''
        B = self.batch_size
        device = self.device

        mask = torch.rand((B,), device=device) < prob
        n = int(mask.sum().item())
        if n == 0:
            return
        
        # Controlled route: fly approximately along +X
        self.p[mask] = torch.tensor([0.0, 0.0, 1.0], device=device) + \
            torch.randn((n , 3), device=device) * 0.05
        self.p_target[mask] = torch.tensor([8.0, 0.0, 1.0], device=device) + \
            torch.randn((n , 3), device=device) * 0.05
        
        wall_x = torch.empty((B, ), device=device).uniform_(3.5, 4.5)
        wall_y = torch.empty((B, ), device=device).uniform_(-0.15, 0.15)

        wall_rx = torch.empty((B, ), device=device).uniform_(0.10, 0.18)
        wall_ry = torch.empty((B, ), device=device).uniform_(4.5, 6.5)

        wall_bottom_z = torch.full((B, ), -1.0, device=device)
        wall_top_z = torch.empty((B, ), device=device).uniform_(1.45, 1.95)
        wall_center_z = 0.5 * (wall_bottom_z + wall_top_z)
        wall_rz = 0.5 * (wall_top_z - wall_bottom_z)

        #Give the height loss a useful vertical signal for selected samples.
        #The final target becomes slightly higher than the wall top 

        overflight_z = (wall_top_z + torch.empty((B,), device=device).uniform_(0.35, 0.75)).clamp(
        min=self.z_min, max=self.z_max)

        self.p_target[mask, 2] = overflight_z[mask]

        wall = torch.stack([
            wall_x,
            wall_y,
            wall_center_z,
            wall_rx,
            wall_ry,
            wall_rz,
        ], dim=-1).unsqueeze(1)  # [B, 1, 6]

        # Disabled samples receive a far-away wall.
        wall[~mask, :, 0] = -100.0

        self.voxels = torch.cat([self.voxels, wall], dim=1)
        
    @torch.no_grad
    def _add_edge_gap_scenario(self, prob=0.25):
        """
        Adds a wall with a side gap near the left or right border of the depth map. 
        The central part of the depth map is blocked.

        self.voxels format: [cx, cy, cz, rx, ry, rz]
        """

        B = self.batch_size 
        device = self.device

        mask = torch.rand((B, ), device=device) < prob
        mask = torch.rand((B,), device=device) < prob
        n = int(mask.sum().item())
        if n == 0:
            return

        # Controlled route: the target is behind the wall, near the image center.
        self.p[mask] = torch.tensor([0.0, 0.0, 1.0], device=device) + \
            torch.randn((n, 3), device=device) * 0.05
        self.p_target[mask] = torch.tensor([8.0, 0.0, 1.0], device=device) + \
            torch.randn((n, 3), device=device) * 0.05

        # Randomize target height only if z randomization is enabled.
        # Otherwise keep the original z=1.0 behavior for this scenario.
        if self.random_z:
            target_z = torch.empty((B,), device=device).uniform_(self.z_min, min(self.z_max, 2.2))
            self.p_target[mask, 2] = target_z[mask]

        wall_x = torch.empty((B,), device=device).uniform_(3.6, 4.4)
        wall_rx = torch.empty((B,), device=device).uniform_(0.10, 0.18)

        # Tall enough to make going around the edge preferable to flying over it.
        wall_bottom_z = torch.full((B,), -1.0, device=device)
        wall_top_z = torch.empty((B,), device=device).uniform_(2.7, 3.8)
        wall_center_z = 0.5 * (wall_bottom_z + wall_top_z)
        wall_rz = 0.5 * (wall_top_z - wall_bottom_z)

        wall_half_y = torch.empty((B,), device=device).uniform_(4.5, 6.0)

        # Randomly select left or right edge.
        side = torch.where(
            torch.rand((B,), device=device) < 0.5,
            torch.tensor(-1.0, device=device),
            torch.tensor(1.0, device=device),
        )

        # Place the gap close to the horizontal FOV edge.
        # For wall_x ~= 4 and fov_x_half_tan ~= 0.53,
        # y ~= 2.1 corresponds to a near-border ray.
        edge_fraction = torch.empty((B,), device=device).uniform_(0.82, 0.95)
        gap_center_y = side * wall_x * self._fov_x_half_tan * edge_fraction
        gap_half_width = torch.empty((B,), device=device).uniform_(0.30, 0.50)

        y_min = -wall_half_y
        y_max = wall_half_y

        block1_y_min = y_min
        block1_y_max = gap_center_y - gap_half_width
        block2_y_min = gap_center_y + gap_half_width
        block2_y_max = y_max

        block1_center_y = 0.5 * (block1_y_min + block1_y_max)
        block2_center_y = 0.5 * (block2_y_min + block2_y_max)
        block1_ry = (0.5 * (block1_y_max - block1_y_min)).clamp_min(0.05)
        block2_ry = (0.5 * (block2_y_max - block2_y_min)).clamp_min(0.05)

        block1 = torch.stack([
            wall_x,
            block1_center_y,
            wall_center_z,
            wall_rx,
            block1_ry,
            wall_rz,
        ], dim=-1)

        block2 = torch.stack([
            wall_x,
            block2_center_y,
            wall_center_z,
            wall_rx,
            block2_ry,
            wall_rz,
        ], dim=-1)

        edge_wall = torch.stack([block1, block2], dim=1)  # [B, 2, 6]

        # Disabled samples receive far-away blocks.
        edge_wall[~mask, :, 0] = -100.0

        self.voxels = torch.cat([self.voxels, edge_wall], dim=1)


    @staticmethod
    @torch.no_grad()
    def update_state_vec(R, a_thr, v_pred, alpha, yaw_inertia=5):
        self_forward_vec = R[..., 0]
        g_std = torch.tensor([0, 0, -9.80665], device=R.device)
        a_thr = a_thr - g_std
        thrust = torch.norm(a_thr, 2, -1, True)
        self_up_vec = a_thr / thrust
        forward_vec = self_forward_vec * yaw_inertia + v_pred
        forward_vec = self_forward_vec * alpha + F.normalize(forward_vec, 2, -1) * (1 - alpha)
        forward_vec[:, 2] = (forward_vec[:, 0] * self_up_vec[:, 0] + forward_vec[:, 1] * self_up_vec[:, 1]) / -self_up_vec[2]
        self_forward_vec = F.normalize(forward_vec, 2, -1)
        self_left_vec = torch.cross(self_up_vec, self_forward_vec)
        return torch.stack([
            self_forward_vec,
            self_left_vec,
            self_up_vec,
        ], -1)

    def render(self, ctl_dt):
        canvas = torch.empty((self.batch_size, self.height, self.width), device=self.device)
        # assert canvas.is_contiguous()
        # assert nearest_pt.is_contiguous()
        # assert self.balls.is_contiguous()
        # assert self.cyl.is_contiguous()
        # assert self.voxels.is_contiguous()
        # assert Rt.is_contiguous()
        quadsim_cuda.render(canvas, self.flow, self.balls, self.cyl, self.cyl_h,
                            self.voxels, self.R @ self.R_cam, self.R_old, self.p,
                            self.p_old, self.drone_radius, self.n_drones_per_group,
                            self._fov_x_half_tan)
        return canvas, None

    def find_vec_to_nearest_pt(self):
        p = self.p + self.v * self.sub_div
        nearest_pt = torch.empty_like(p)
        quadsim_cuda.find_nearest_pt(nearest_pt, self.balls, self.cyl, self.cyl_h, self.voxels, p, self.drone_radius, self.n_drones_per_group)
        return nearest_pt - p

    def run(self, act_pred, ctl_dt=1/15, v_pred=None):
        self.dg = self.dg * math.sqrt(1 - ctl_dt / 4) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt / 4)
        self.p_old = self.p
        self.act, self.p, self.v, self.a = run(
            self.R, self.dg, self.z_drag_coef, self.drag_2, self.pitch_ctl_delay,
            act_pred, self.act, self.p, self.v, self.v_wind, self.a,
            self.grad_decay, ctl_dt, 0.5)
        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)

    def _run(self, act_pred, ctl_dt=1/15, v_pred=None):
        alpha = torch.exp(-self.pitch_ctl_delay * ctl_dt)
        self.act = act_pred * (1 - alpha) + self.act * alpha
        self.dg = self.dg * math.sqrt(1 - ctl_dt) + torch.randn_like(self.dg) * 0.2 * math.sqrt(ctl_dt)
        z_drag = 0
        if self.z_drag_coef is not None:
            v_up = torch.sum(self.v * self.R[..., 2], -1, keepdim=True) * self.R[..., 2]
            v_prep = self.v - v_up
            motor_velocity = (self.act - self.g_std).norm(2, -1, True).sqrt()
            z_drag = self.z_drag_coef * v_prep * motor_velocity * 0.07
        drag = self.drag_2 * self.v * self.v.norm(2, -1, True)
        a_next = self.act + self.dg - z_drag - drag
        self.p_old = self.p
        self.p = g_decay(self.p, self.grad_decay ** ctl_dt) + self.v * ctl_dt + 0.5 * self.a * ctl_dt**2
        self.v = g_decay(self.v, self.grad_decay ** ctl_dt) + (self.a + a_next) / 2 * ctl_dt
        self.a = a_next

        # update attitude
        alpha = torch.exp(-self.yaw_ctl_delay * ctl_dt)
        self.R_old = self.R.clone()
        self.R = quadsim_cuda.update_state_vec(self.R, self.act, v_pred, alpha, 5)

