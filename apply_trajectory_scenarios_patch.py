#!/usr/bin/env python3
"""Patch DiffPhysDrone_v2 feat/trajectory with mixed reference scenarios.

Run from the repository root:
    python apply_trajectory_scenarios_patch.py

The script creates one-time .before_traj_scenarios backups and validates
the resulting Python syntax before writing files.
"""
from __future__ import annotations

import argparse
import ast
from pathlib import Path
import re
import shutil
import sys


TRAJ_CHOICES = (
    "legacy",
    "mixed",
    "straight",
    "smooth_turn",
    "sharp_yaw_90",
    "s_curve",
    "climb_over_wall",
    "descend_after_obstacle",
    "side_gap",
    "edge_gap",
    "around_obstacle",
    "time_shifted_crossing",
)

CLASS_CONSTANT = 'class Env:\n    TRAJ_SCENARIOS = (\n        "straight",\n        "smooth_turn",\n        "sharp_yaw_90",\n        "s_curve",\n        "climb_over_wall",\n        "descend_after_obstacle",\n        "side_gap",\n        "edge_gap",\n        "around_obstacle",\n        "time_shifted_crossing",\n    )\n'
ATTRS_BLOCK = '\n        self.traj_scenario = str(traj_scenario)\n        self.traj_waypoints = max(5, int(traj_waypoints))\n        default_traj_probs = [\n            0.15,  # straight\n            0.10,  # smooth_turn\n            0.08,  # sharp_yaw_90\n            0.10,  # s_curve\n            0.10,  # climb_over_wall\n            0.10,  # descend_after_obstacle\n            0.10,  # side_gap\n            0.08,  # edge_gap\n            0.09,  # around_obstacle\n            0.10,  # time_shifted_crossing\n        ]\n        if traj_scenario_probs is None:\n            parsed_probs = default_traj_probs\n        elif isinstance(traj_scenario_probs, str):\n            parsed_probs = [\n                float(value.strip())\n                for value in traj_scenario_probs.split(",")\n                if value.strip()\n            ]\n        else:\n            parsed_probs = [float(value) for value in traj_scenario_probs]\n\n        if len(parsed_probs) != len(self.TRAJ_SCENARIOS):\n            raise ValueError(\n                "traj_scenario_probs must contain exactly "\n                f"{len(self.TRAJ_SCENARIOS)} values"\n            )\n        self.traj_scenario_probs = torch.tensor(\n            parsed_probs, device=device, dtype=torch.float32\n        )\n        if torch.any(self.traj_scenario_probs < 0):\n            raise ValueError("traj_scenario_probs must be non-negative")\n        prob_sum = self.traj_scenario_probs.sum()\n        if float(prob_sum) <= 0:\n            raise ValueError("traj_scenario_probs sum must be positive")\n        self.traj_scenario_probs /= prob_sum\n'
METHODS_BLOCK = '\n    @torch.no_grad()\n    def _build_reference_trajectory(self):\n        """\n        Build the reference path used by trajectory conditioning.\n\n        traj_scenario="legacy" preserves the previous straight p->p_target\n        behaviour. "mixed" samples one of TRAJ_SCENARIOS per batch element.\n        A concrete scenario name forces that scenario for the whole batch.\n        """\n        B = self.batch_size\n        device = self.device\n        dtype = self.p.dtype\n        M = self.traj_waypoints\n\n        if self.traj_scenario == "legacy":\n            u = torch.linspace(0.0, 1.0, M, device=device, dtype=dtype)\n            start = self.p.detach().clone()\n            goal = self.p_target.detach().clone()\n            waypoints = (\n                start[:, None, :]\n                + u[None, :, None] * (goal - start)[:, None, :]\n            )\n            scenario_id = torch.full(\n                (B,), -1, device=device, dtype=torch.long\n            )\n            self._set_reference_trajectory(waypoints, scenario_id)\n            return\n\n        if self.traj_scenario == "mixed":\n            probs = self.traj_scenario_probs.to(device=device, dtype=dtype)\n            scenario_id = torch.multinomial(probs, B, replacement=True)\n        else:\n            if self.traj_scenario not in self.TRAJ_SCENARIOS:\n                allowed = ", ".join(("legacy", "mixed", *self.TRAJ_SCENARIOS))\n                raise ValueError(\n                    f"Unknown traj_scenario={self.traj_scenario!r}. "\n                    f"Expected one of: {allowed}"\n                )\n            scenario_id = torch.full(\n                (B,),\n                self.TRAJ_SCENARIOS.index(self.traj_scenario),\n                device=device,\n                dtype=torch.long,\n            )\n\n        self.traj_scenario_id = scenario_id\n        start = self.p.detach().clone()\n\n        # Vertical obstacle scenarios need a controlled initial altitude;\n        # otherwise random_z could place the drone already above the wall.\n        vertical_mask = (scenario_id == 4) | (scenario_id == 5)\n        if vertical_mask.any():\n            start[vertical_mask, 2] = torch.empty(\n                (int(vertical_mask.sum().item()),),\n                device=device,\n                dtype=dtype,\n            ).uniform_(0.8, 1.2)\n\n        # Build every path in a local frame. Wall/gap scenarios remain aligned\n        # with +X because quadsim voxels are axis-aligned boxes.\n        if self.random_rotation and self.traj_scenario == "mixed":\n            base_yaw = torch.empty(\n                (B,), device=device, dtype=dtype\n            ).uniform_(-math.pi, math.pi)\n            axis_aligned = (\n                (scenario_id == 4)\n                | (scenario_id == 5)\n                | (scenario_id == 6)\n                | (scenario_id == 7)\n            )\n            base_yaw[axis_aligned] = 0.0\n        else:\n            base_yaw = torch.zeros((B,), device=device, dtype=dtype)\n\n        u = torch.linspace(0.0, 1.0, M, device=device, dtype=dtype)\n        x = torch.zeros((B, M), device=device, dtype=dtype)\n        y = torch.zeros_like(x)\n        z_offset = torch.zeros_like(x)\n\n        length = torch.empty(\n            (B,), device=device, dtype=dtype\n        ).uniform_(7.0, 9.0)\n        x[:] = length[:, None] * u[None, :]\n\n        # 1. straight trajectory: default x/y/z values.\n\n        # 2. smooth turn by 30-45 degrees.\n        mask = scenario_id == 1\n        if mask.any():\n            n = int(mask.sum().item())\n            sign = torch.where(\n                torch.rand((n,), device=device) < 0.5,\n                -torch.ones((n,), device=device, dtype=dtype),\n                torch.ones((n,), device=device, dtype=dtype),\n            )\n            angle = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(30.0, 45.0) * math.pi / 180.0\n            radius = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(7.0, 10.0)\n            theta = angle[:, None] * u[None, :]\n            x[mask] = radius[:, None] * torch.sin(theta)\n            y[mask] = (\n                sign[:, None]\n                * radius[:, None]\n                * (1.0 - torch.cos(theta))\n            )\n\n        # 3. sharp yaw turn by 90 degrees.\n        mask = scenario_id == 2\n        if mask.any():\n            n = int(mask.sum().item())\n            sign = torch.where(\n                torch.rand((n,), device=device) < 0.5,\n                -torch.ones((n,), device=device, dtype=dtype),\n                torch.ones((n,), device=device, dtype=dtype),\n            )\n            radius = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(1.5, 2.3)\n            entry = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(2.0, 3.0)\n            exit_len = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(3.0, 4.5)\n\n            s_path = u[None, :].expand(n, M)\n            first = s_path <= 0.25\n            arc = (s_path > 0.25) & (s_path <= 0.75)\n            last = s_path > 0.75\n\n            x_m = torch.zeros((n, M), device=device, dtype=dtype)\n            y_m = torch.zeros_like(x_m)\n\n            q = (s_path / 0.25).clamp(0.0, 1.0)\n            x_m = torch.where(first, entry[:, None] * q, x_m)\n\n            arc_u = ((s_path - 0.25) / 0.50).clamp(0.0, 1.0)\n            theta = 0.5 * math.pi * arc_u\n            arc_x = entry[:, None] + radius[:, None] * torch.sin(theta)\n            arc_y = (\n                sign[:, None]\n                * radius[:, None]\n                * (1.0 - torch.cos(theta))\n            )\n            x_m = torch.where(arc, arc_x, x_m)\n            y_m = torch.where(arc, arc_y, y_m)\n\n            exit_u = ((s_path - 0.75) / 0.25).clamp(0.0, 1.0)\n            final_x = entry + radius\n            final_y = sign * radius\n            exit_x = final_x[:, None].expand_as(s_path)\n            exit_y = (\n                final_y[:, None]\n                + sign[:, None] * exit_len[:, None] * exit_u\n            )\n            x_m = torch.where(last, exit_x, x_m)\n            y_m = torch.where(last, exit_y, y_m)\n\n            x[mask] = x_m\n            y[mask] = y_m\n\n        # 4. S-curve.\n        mask = scenario_id == 3\n        if mask.any():\n            n = int(mask.sum().item())\n            length_m = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(8.0, 10.0)\n            amplitude = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(1.0, 1.8)\n            sign = torch.where(\n                torch.rand((n,), device=device) < 0.5,\n                -torch.ones((n,), device=device, dtype=dtype),\n                torch.ones((n,), device=device, dtype=dtype),\n            )\n            x[mask] = length_m[:, None] * u[None, :]\n            y[mask] = (\n                sign[:, None]\n                * amplitude[:, None]\n                * torch.sin(2.0 * math.pi * u[None, :])\n            )\n\n        # 5. climb over wall and remain above it.\n        mask = scenario_id == 4\n        if mask.any():\n            n = int(mask.sum().item())\n            wall_x = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(3.6, 4.4)\n            wall_top = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(2.0, 2.6)\n            clearance = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(0.45, 0.75)\n            length_m = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(7.5, 9.0)\n            x_m = length_m[:, None] * u[None, :]\n            x[mask] = x_m\n            high = wall_top + clearance\n            climb_u = (\n                (x_m - (wall_x[:, None] - 2.2)) / 3.0\n            ).clamp(0.0, 1.0)\n            smooth = climb_u * climb_u * (3.0 - 2.0 * climb_u)\n            z_offset[mask] = (\n                high[:, None] - start[mask, 2][:, None]\n            ) * smooth\n\n            wall_x_world = start[mask, 0] + wall_x\n            wall_y_world = start[mask, 1]\n            self._append_wall(\n                mask,\n                wall_x_world,\n                wall_y_world,\n                wall_top,\n                half_width_y=5.5,\n            )\n\n        # 6. climb and explicitly descend after the obstacle.\n        mask = scenario_id == 5\n        if mask.any():\n            n = int(mask.sum().item())\n            wall_x = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(3.4, 4.0)\n            wall_top = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(1.9, 2.4)\n            clearance = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(0.45, 0.70)\n            length_m = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(8.0, 9.5)\n            x_m = length_m[:, None] * u[None, :]\n            x[mask] = x_m\n            high = wall_top + clearance\n            dz = high[:, None] - start[mask, 2][:, None]\n\n            climb_u = (\n                (x_m - (wall_x[:, None] - 2.0)) / 2.2\n            ).clamp(0.0, 1.0)\n            climb = climb_u * climb_u * (3.0 - 2.0 * climb_u)\n            descent_u = (\n                (x_m - (wall_x[:, None] + 0.8)) / 2.8\n            ).clamp(0.0, 1.0)\n            descent = (\n                descent_u\n                * descent_u\n                * (3.0 - 2.0 * descent_u)\n            )\n            z_offset[mask] = dz * (climb - descent)\n\n            wall_x_world = start[mask, 0] + wall_x\n            wall_y_world = start[mask, 1]\n            self._append_wall(\n                mask,\n                wall_x_world,\n                wall_y_world,\n                wall_top,\n                half_width_y=5.5,\n            )\n\n        # 7. side gap trajectory: moderate lateral displacement.\n        mask = scenario_id == 6\n        if mask.any():\n            n = int(mask.sum().item())\n            sign = torch.where(\n                torch.rand((n,), device=device) < 0.5,\n                -torch.ones((n,), device=device, dtype=dtype),\n                torch.ones((n,), device=device, dtype=dtype),\n            )\n            wall_x = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(3.7, 4.3)\n            gap_y = sign * torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(0.9, 1.35)\n            gap_half = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(0.45, 0.65)\n            length_m = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(7.5, 9.0)\n            x[mask] = length_m[:, None] * u[None, :]\n            y[mask] = gap_y[:, None] * torch.sin(\n                math.pi * u[None, :]\n            )\n\n            wall_x_world = start[mask, 0] + wall_x\n            gap_y_world = start[mask, 1] + gap_y\n            self._append_gap_wall(\n                mask,\n                wall_x_world,\n                gap_y_world,\n                gap_half,\n                half_width_y=3.5,\n                wall_center_y=start[mask, 1],\n            )\n\n        # 8. edge gap trajectory: gap near the depth-map border.\n        mask = scenario_id == 7\n        if mask.any():\n            n = int(mask.sum().item())\n            sign = torch.where(\n                torch.rand((n,), device=device) < 0.5,\n                -torch.ones((n,), device=device, dtype=dtype),\n                torch.ones((n,), device=device, dtype=dtype),\n            )\n            wall_x = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(3.8, 4.5)\n            gap_y = sign * torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(1.55, 1.90)\n            gap_half = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(0.38, 0.52)\n            length_m = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(7.5, 9.0)\n            x[mask] = length_m[:, None] * u[None, :]\n            y[mask] = gap_y[:, None] * torch.sin(\n                math.pi * u[None, :]\n            )\n\n            wall_x_world = start[mask, 0] + wall_x\n            gap_y_world = start[mask, 1] + gap_y\n            self._append_gap_wall(\n                mask,\n                wall_x_world,\n                gap_y_world,\n                gap_half,\n                half_width_y=3.6,\n                wall_center_y=start[mask, 1],\n            )\n\n        # 9. smooth trajectory around a central obstacle.\n        mask = scenario_id == 8\n        if mask.any():\n            n = int(mask.sum().item())\n            sign = torch.where(\n                torch.rand((n,), device=device) < 0.5,\n                -torch.ones((n,), device=device, dtype=dtype),\n                torch.ones((n,), device=device, dtype=dtype),\n            )\n            obstacle_x = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(3.5, 4.5)\n            offset = sign * torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(1.1, 1.7)\n            length_m = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(7.5, 9.0)\n            x[mask] = length_m[:, None] * u[None, :]\n            y[mask] = offset[:, None] * torch.sin(\n                math.pi * u[None, :]\n            )\n\n            c_m = torch.cos(base_yaw[mask])\n            s_m = torch.sin(base_yaw[mask])\n            obs_x_world = (\n                start[mask, 0] + c_m * obstacle_x\n            )\n            obs_y_world = (\n                start[mask, 1] + s_m * obstacle_x\n            )\n            self._append_central_obstacle(\n                mask,\n                obs_x_world,\n                obs_y_world,\n                start[mask, 2],\n            )\n\n        # 10. same geometric crossing path with a randomized temporal shift.\n        # This prepares the interface for a future moving multi-agent obstacle.\n        mask = scenario_id == 9\n        if mask.any():\n            n = int(mask.sum().item())\n            length_m = torch.empty(\n                (n,), device=device, dtype=dtype\n            ).uniform_(7.5, 9.0)\n            x[mask] = length_m[:, None] * u[None, :]\n            y[mask] = 0.0\n\n        c = torch.cos(base_yaw)\n        s = torch.sin(base_yaw)\n        world_x = c[:, None] * x - s[:, None] * y\n        world_y = s[:, None] * x + c[:, None] * y\n        waypoints = torch.stack(\n            [\n                start[:, None, 0] + world_x,\n                start[:, None, 1] + world_y,\n                start[:, None, 2] + z_offset,\n            ],\n            dim=-1,\n        )\n\n        start_delay = torch.zeros((B,), device=device, dtype=dtype)\n        crossing_mask = scenario_id == 9\n        if crossing_mask.any():\n            start_delay[crossing_mask] = torch.empty(\n                (int(crossing_mask.sum().item()),),\n                device=device,\n                dtype=dtype,\n            ).uniform_(0.4, 1.8)\n\n        self.p = waypoints[:, 0].clone()\n        self.p_target = waypoints[:, -1].clone()\n        self._set_reference_trajectory(\n            waypoints,\n            scenario_id,\n            start_delay=start_delay,\n        )\n\n    @torch.no_grad()\n    def _set_reference_trajectory(\n        self,\n        waypoints,\n        scenario_id,\n        start_delay=None,\n    ):\n        """Finalize timing and tangent yaw for already generated waypoints."""\n        B = self.batch_size\n        device = self.device\n        dtype = self.p.dtype\n\n        seg = waypoints[:, 1:] - waypoints[:, :-1]\n        seg_len = torch.norm(seg, dim=-1).clamp_min(1e-4)\n        speed = self.max_speed.reshape(-1).clamp_min(0.5)\n        seg_dt = (seg_len / speed[:, None]).clamp_min(0.10)\n\n        traj_times = torch.cat(\n            [\n                torch.zeros((B, 1), device=device, dtype=dtype),\n                torch.cumsum(seg_dt, dim=1),\n            ],\n            dim=1,\n        )\n        if start_delay is None:\n            start_delay = torch.zeros((B,), device=device, dtype=dtype)\n        traj_times[:, 1:] += start_delay[:, None]\n\n        self.traj_scenario_id = scenario_id\n        self.traj_waypoints_world = waypoints\n        self.traj_times = traj_times\n        self.traj_total_time = traj_times[:, -1].clamp_min(1.0)\n        self.traj_seg_yaw = torch.atan2(seg[..., 1], seg[..., 0])\n        self.traj_initial_direction = F.normalize(seg[:, 0], dim=-1)\n        self.traj_start_delay = start_delay\n\n        crossing_index = self.traj_waypoints // 2\n        self.traj_crossing_time = torch.full(\n            (B,), float("nan"), device=device, dtype=dtype\n        )\n        crossing_mask = scenario_id == 9\n        if crossing_mask.any():\n            self.traj_crossing_time[crossing_mask] = (\n                traj_times[crossing_mask, crossing_index]\n            )\n\n    @torch.no_grad()\n    def _append_wall(\n        self,\n        mask,\n        wall_x_active,\n        wall_y_active,\n        wall_top_active,\n        half_width_y,\n    ):\n        """Append one full wall; inactive entries get a far-away box."""\n        B = self.batch_size\n        device = self.device\n        dtype = self.p.dtype\n        wall = torch.zeros((B, 1, 6), device=device, dtype=dtype)\n        wall[:, :, 0] = -100.0\n\n        bottom = -0.5\n        center_z = 0.5 * (bottom + wall_top_active)\n        half_z = 0.5 * (wall_top_active - bottom)\n\n        wall[mask, 0, 0] = wall_x_active\n        wall[mask, 0, 1] = wall_y_active\n        wall[mask, 0, 2] = center_z\n        wall[mask, 0, 3] = 0.16\n        wall[mask, 0, 4] = half_width_y\n        wall[mask, 0, 5] = half_z\n        self.voxels = torch.cat([self.voxels, wall], dim=1)\n\n    @torch.no_grad()\n    def _append_gap_wall(\n        self,\n        mask,\n        wall_x_active,\n        gap_y_active,\n        gap_half_active,\n        half_width_y,\n        wall_center_y,\n    ):\n        """Append two boxes forming a vertical wall with a lateral gap."""\n        B = self.batch_size\n        device = self.device\n        dtype = self.p.dtype\n        wall = torch.zeros((B, 2, 6), device=device, dtype=dtype)\n        wall[:, :, 0] = -100.0\n\n        y_min = wall_center_y - half_width_y\n        y_max = wall_center_y + half_width_y\n        left_min = y_min\n        left_max = gap_y_active - gap_half_active\n        right_min = gap_y_active + gap_half_active\n        right_max = y_max\n\n        center_z = torch.full_like(gap_y_active, 1.5)\n        half_z = torch.full_like(gap_y_active, 2.5)\n\n        wall[mask, 0, 0] = wall_x_active\n        wall[mask, 0, 1] = 0.5 * (left_min + left_max)\n        wall[mask, 0, 2] = center_z\n        wall[mask, 0, 3] = 0.16\n        wall[mask, 0, 4] = (\n            0.5 * (left_max - left_min)\n        ).clamp_min(0.05)\n        wall[mask, 0, 5] = half_z\n\n        wall[mask, 1, 0] = wall_x_active\n        wall[mask, 1, 1] = 0.5 * (right_min + right_max)\n        wall[mask, 1, 2] = center_z\n        wall[mask, 1, 3] = 0.16\n        wall[mask, 1, 4] = (\n            0.5 * (right_max - right_min)\n        ).clamp_min(0.05)\n        wall[mask, 1, 5] = half_z\n        self.voxels = torch.cat([self.voxels, wall], dim=1)\n\n    @torch.no_grad()\n    def _append_central_obstacle(\n        self,\n        mask,\n        obstacle_x_active,\n        obstacle_y_active,\n        z_active,\n    ):\n        """Append a box obstacle centered on the direct route."""\n        B = self.batch_size\n        device = self.device\n        dtype = self.p.dtype\n        obstacle = torch.zeros((B, 1, 6), device=device, dtype=dtype)\n        obstacle[:, :, 0] = -100.0\n\n        obstacle[mask, 0, 0] = obstacle_x_active\n        obstacle[mask, 0, 1] = obstacle_y_active\n        obstacle[mask, 0, 2] = z_active\n        obstacle[mask, 0, 3] = 0.65\n        obstacle[mask, 0, 4] = 0.65\n        obstacle[mask, 0, 5] = 0.90\n        self.voxels = torch.cat([self.voxels, obstacle], dim=1)\n\n    @torch.no_grad()\n    def _sample_reference(self, query_time):\n        """\n        Interpolate the piecewise-linear trajectory.\n\n        query_time: [B, K], seconds from episode start.\n        Returns p_ref [B,K,3] and yaw_ref [B,K].\n        """\n        B, K = query_time.shape\n        M = self.traj_waypoints_world.shape[1]\n\n        seg_idx = torch.searchsorted(\n            self.traj_times.contiguous(),\n            query_time.contiguous(),\n            right=True,\n        ) - 1\n        seg_idx = seg_idx.clamp(0, M - 2)\n\n        t0 = torch.gather(self.traj_times, 1, seg_idx)\n        t1 = torch.gather(self.traj_times, 1, seg_idx + 1)\n        alpha = (\n            (query_time - t0) / (t1 - t0).clamp_min(1e-4)\n        ).clamp(0.0, 1.0)\n\n        idx3 = seg_idx[..., None].expand(B, K, 3)\n        p0 = torch.gather(self.traj_waypoints_world, 1, idx3)\n        p1 = torch.gather(self.traj_waypoints_world, 1, idx3 + 1)\n        p_ref = p0 + alpha[..., None] * (p1 - p0)\n\n        yaw_ref = torch.gather(self.traj_seg_yaw, 1, seg_idx)\n        return p_ref, yaw_ref\n\n    @torch.no_grad()\n    def get_reference_velocity(self, sim_time, lookahead=None):\n        """\n        World-frame velocity command tangent to the timed reference path.\n\n        main_cuda.py should use this instead of p_target-p when a generated\n        trajectory scenario is active.\n        """\n        if lookahead is None:\n            lookahead = max(self.traj_dt, 0.25)\n\n        query_time = torch.full(\n            (self.batch_size, 1),\n            float(sim_time) + float(lookahead),\n            device=self.device,\n            dtype=self.p.dtype,\n        )\n        p_ref, _ = self._sample_reference(query_time)\n        delta = p_ref[:, 0] - self.p.detach()\n\n        norm = torch.norm(delta, dim=-1, keepdim=True)\n        direction = delta / norm.clamp_min(1e-4)\n        speed = torch.minimum(\n            norm / max(float(lookahead), 1e-3),\n            self.max_speed,\n        )\n        return direction * speed\n\n    @torch.no_grad()\n    def get_traj_features(self, sim_time, R_local):\n        """\n        Return [B,K,6]:\n        [dt, dx_body, dy_body, dz_body, sin(dtheta), cos(dtheta)].\n        """\n        B = self.batch_size\n        device = self.device\n        dtype = self.p.dtype\n        K = self.traj_points\n\n        step_ids = torch.arange(1, K + 1, device=device, dtype=dtype)\n        dt = step_ids[None, :] * self.traj_dt\n        query_time = torch.full(\n            (B, 1), float(sim_time), device=device, dtype=dtype\n        ) + dt\n\n        p_ref, yaw_ref = self._sample_reference(query_time)\n        dp_world = p_ref - self.p.detach()[:, None, :]\n        dp_local = torch.bmm(dp_world, R_local)\n\n        fwd = R_local[:, :, 0]\n        yaw_now = torch.atan2(fwd[:, 1], fwd[:, 0])\n        dtheta = yaw_ref - yaw_now[:, None]\n        dtheta = torch.atan2(torch.sin(dtheta), torch.cos(dtheta))\n\n        return torch.stack(\n            [\n                dt.expand(B, K) / self.traj_time_scale,\n                dp_local[:, :, 0] / self.traj_pos_scale,\n                dp_local[:, :, 1] / self.traj_pos_scale,\n                dp_local[:, :, 2] / self.traj_pos_scale,\n                torch.sin(dtheta),\n                torch.cos(dtheta),\n            ],\n            dim=-1,\n        )\n'


def replace_once(text: str, pattern: str, replacement: str, label: str) -> str:
    updated, count = re.subn(pattern, replacement, text, count=1, flags=re.S)
    if count != 1:
        raise RuntimeError(
            f"Could not patch {label}: expected one match, found {count}"
        )
    return updated


def patch_env(text: str) -> str:
    if "TRAJ_SCENARIOS = (" in text:
        raise RuntimeError("env_cuda.py already contains the trajectory patch")

    text = replace_once(
        text,
        r"class Env:\s*\n",
        CLASS_CONSTANT,
        "Env.TRAJ_SCENARIOS",
    )

    text = replace_once(
        text,
        (
            r"traj_points=8,\s*traj_dt=0\.25,\s*"
            r"traj_pos_scale=5\.0,\s*traj_time_scale=2\.0,\s*"
            r"(?=\)\s*->\s*None:)"
        ),
        (
            "traj_points=8, traj_dt=0.25, traj_pos_scale=5.0, "
            "traj_time_scale=2.0, traj_scenario='legacy', "
            "traj_scenario_probs=None, traj_waypoints=9, "
        ),
        "Env.__init__ trajectory arguments",
    )

    text = replace_once(
        text,
        r"(\n\s*self\.traj_time_scale\s*=\s*traj_time_scale\s*\n)",
        r"\1" + ATTRS_BLOCK,
        "Env.__init__ trajectory attributes",
    )

    text = replace_once(
        text,
        (
            r"\n    @torch\.no_grad\(\)\n"
            r"    def _build_reference_trajectory\(self\):.*?"
            r"(?=\n    @torch\.no_grad\(\)\n"
            r"    def _add_over_wall_scenario)"
        ),
        "\n" + METHODS_BLOCK.rstrip() + "\n",
        "reference trajectory methods",
    )

    orientation_pattern = (
        r"\s*self\.R\s*=\s*quadsim_cuda\.update_state_vec\("
        r"R,\s*self\.act,\s*"
        r"torch\.randn\(\(B,\s*3\),\s*device=device\)\s*\*\s*0\.2"
        r"\s*\+\s*F\.normalize\(self\.p_target\s*-\s*self\.p\),"
        r"\s*torch\.zeros_like\(self\.yaw_ctl_delay\),\s*5\)"
    )
    orientation_replacement = """
        initial_heading = getattr(
            self,
            "traj_initial_direction",
            F.normalize(self.p_target - self.p, dim=-1),
        )
        self.R = quadsim_cuda.update_state_vec(
            R,
            self.act,
            torch.randn((B, 3), device=device) * 0.2 + initial_heading,
            torch.zeros_like(self.yaw_ctl_delay),
            5,
        )"""
    text = replace_once(
        text,
        orientation_pattern,
        orientation_replacement,
        "initial trajectory heading",
    )

    ast.parse(text, filename="env_cuda.py")
    return text


def patch_main(text: str) -> str:
    if "--traj_scenario" in text:
        raise RuntimeError("main_cuda.py already contains the trajectory patch")

    choices_literal = repr(list(TRAJ_CHOICES))
    arg_block = f"""
parser.add_argument(
    '--traj_scenario',
    type=str,
    default='legacy',
    choices={choices_literal},
)
parser.add_argument(
    '--traj_scenario_probs',
    type=str,
    default=None,
    help='10 comma-separated probabilities in TRAJ_SCENARIOS order',
)
parser.add_argument('--traj_waypoints', type=int, default=9)
"""

    text = replace_once(
        text,
        (
            r"(parser\.add_argument\('--traj_time_scale',\s*"
            r"type=float,\s*default=2\.0\)\s*\n)"
        ),
        r"\1" + arg_block,
        "main trajectory CLI arguments",
    )

    text = replace_once(
        text,
        r"(\s*traj_time_scale=args\.traj_time_scale,\s*\n)",
        (
            r"\1"
            "    traj_scenario=args.traj_scenario,\n"
            "    traj_scenario_probs=args.traj_scenario_probs,\n"
            "    traj_waypoints=args.traj_waypoints,\n"
        ),
        "Env trajectory arguments in main",
    )

    target_pattern = (
        r"        if args\.yaw_drift:\s*\n"
        r"            target_v_raw = torch\.squeeze\("
        r"target_v_raw\[:, None\] @ R_drift, 1\)\s*\n"
        r"        else:\s*\n"
        r"            target_v_raw = env\.p_target - env\.p\.detach\(\)"
    )
    target_replacement = """        if args.traj_conditioning and args.traj_scenario != 'legacy':
            target_v_raw = env.get_reference_velocity(sim_time)
        elif args.yaw_drift:
            target_v_raw = torch.squeeze(
                target_v_raw[:, None] @ R_drift, 1
            )
        else:
            target_v_raw = env.p_target - env.p.detach()"""
    text = replace_once(
        text,
        target_pattern,
        target_replacement,
        "reference velocity selection",
    )

    text = replace_once(
        text,
        r"target_v_unit\s*=\s*target_v_raw\s*/\s*target_v_norm",
        "target_v_unit = target_v_raw / target_v_norm.clamp_min(1e-6)",
        "target velocity normalization",
    )

    text = replace_once(
        text,
        (
            r"target_v_history_normalized\s*=\s*"
            r"target_v_history\s*/\s*target_v_history_norm\[\.\.\., None\]"
        ),
        (
            "target_v_history_normalized = target_v_history / "
            "target_v_history_norm.clamp_min(1e-6)[..., None]"
        ),
        "loss target velocity normalization",
    )

    ast.parse(text, filename="main_cuda.py")
    return text


def write_with_backup(path: Path, content: str) -> None:
    backup = path.with_suffix(path.suffix + ".before_traj_scenarios")
    if not backup.exists():
        shutil.copy2(path, backup)
    path.write_text(content, encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", type=Path, default=Path("."))
    parser.add_argument("--check-only", action="store_true")
    args = parser.parse_args()

    env_path = args.repo / "env_cuda.py"
    main_path = args.repo / "main_cuda.py"
    for path in (env_path, main_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    env_new = patch_env(env_path.read_text(encoding="utf-8"))
    main_new = patch_main(main_path.read_text(encoding="utf-8"))

    if args.check_only:
        print("Patch matches both files and resulting syntax is valid.")
        return 0

    write_with_backup(env_path, env_new)
    write_with_backup(main_path, main_new)
    print("Patched env_cuda.py and main_cuda.py")
    print("Backups: *.before_traj_scenarios")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise
