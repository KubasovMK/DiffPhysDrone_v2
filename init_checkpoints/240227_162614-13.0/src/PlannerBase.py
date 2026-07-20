#!/usr/bin/env python3
import collections
import copy
import math
import os
import time
import numpy as np
import rospy
import cv2
import pandas as pd

from cv_bridge import CvBridge, CvBridgeError
from nav_msgs.msg import Odometry
from quadrotor_msgs.msg import TrajectoryPoint
from quadrotor_msgs.msg import Trajectory
from sensor_msgs.msg import Image
from std_msgs.msg import Empty
from scipy.spatial.transform import Rotation as R
from agile_autonomy_msgs.msg import MultiTrajectory
from geometry_msgs.msg import Point
# import tensorflow as tf
import torch
import torch.nn.functional as F

from geometry_msgs.msg import Quaternion
from quadrotor_msgs.msg import ControlCommand
from .rotation import matrix_to_quaternion, quaternion_to_matrix
from .model import Model

torch.set_grad_enabled(False)


class PlanBase(object):
    def __init__(self, config):
        self.config = config
        self.odometry = Odometry()
        self.gt_odometry = Odometry()
        self.maneuver_complete = False
        self.use_network = False
        self.net_initialized = False
        self.reference_initialized = False
        self.rollout_idx = 0
        self.odometry_used_for_inference = None
        self.time_prediction = None
        self.last_depth_received = rospy.Time.now()
        self.reference_progress = 0
        self.reference_len = 1
        self.bridge = CvBridge()
        self.n_times_expert = 0
        self.n_times_net = 0.00001
        self.start_time = None
        self.quad_name = config.quad_name
        self.depth_topic = config.depth_topic
        self.rgb_topic = config.rgb_topic
        self.odometry_topic = config.odometry_topic

        if config.no_odom:
            self.model = Model(4 + 3, 6).eval()
        else:
            self.model = Model(7 + 3, 6).eval()
        state_dict = torch.load(config.net_weight, map_location='cpu')
        # state_dict['fc.weight'] = state_dict['fc.weight'].reshape(3, -1, self.model.fc.in_features)[:, :2].reshape(6, -1)
        self.model.load_state_dict(state_dict, strict=False)

        # Init Network
        self.model(torch.zeros(1, 1, 12, 16), torch.zeros(1, self.model.dim_obs))
        print("Net initialized")
        self.net_initialized = True
        self.ground_truth_odom = rospy.Subscriber(self.odometry_topic,
                                                  Odometry,
                                                  self.callback_gt_odometry,
                                                  queue_size=1,
                                                  tcp_nodelay=True)
        if config.use_depth:
            self.depth_sub = rospy.Subscriber(self.depth_topic, Image,
                                              self.callback_depth, queue_size=1)
        self.land_sub = rospy.Subscriber("/" + config.quad_name + "/autopilot/land",
                                         Empty, self.callback_land, queue_size=1)
        self.timer_input = rospy.Timer(rospy.Duration(1. / config.input_update_freq),
                                       self.update_input_queues)
        # self.timer_net = rospy.Timer(rospy.Duration(1. / config.network_frequency),
        #                              self._generate_plan)
        self.cmd_pub = rospy.Publisher("/" + self.quad_name + "/autopilot/control_command_input", ControlCommand, queue_size=1)
        self.a_set_pub = rospy.Publisher('~a_set', Point, queue_size=10)
        self.v_pub = rospy.Publisher('~v', Point, queue_size=10)

        self.odometry = None
        # self.odom_q = collections.deque()
        self.h = None
        self.forward = None
        self.p_target = None
        self.plan_until_t = 0
        self.margin = torch.tensor([config.margin])

    def start_planning(self):
        self.plan_until_t = time.time() + 100
        self.h = None
        self.forward = None
        print('start planning')
        target_pos = self.full_reference[-1].pose.position
        self.p_target = torch.as_tensor(
            [target_pos.x, target_pos.y, target_pos.z], dtype=torch.float)

    def load_trajectory(self, traj_fname):
        self.reference_initialized = False
        traj_df = pd.read_csv(traj_fname, delimiter=',')
        self.reference_len = traj_df.shape[0]
        self.full_reference = Trajectory()
        time = ['time_from_start']
        time_values = traj_df[time].values
        pos = ["pos_x", "pos_y", "pos_z"]
        pos_values = traj_df[pos].values
        vel = ["vel_x", "vel_y", "vel_z"]
        vel_values = traj_df[vel].values
        for i in range(self.reference_len):
            point = TrajectoryPoint()
            point.time_from_start = rospy.Duration(time_values[i])
            point.pose.position.x = pos_values[i][0]
            point.pose.position.y = pos_values[i][1]
            point.pose.position.z = pos_values[i][2]
            point.velocity.linear.x = vel_values[i][0]
            point.velocity.linear.y = vel_values[i][1]
            point.velocity.linear.z = vel_values[i][2]
            self.full_reference.points.append(point)
        # Change type for easier use
        self.full_reference = self.full_reference.points
        self.reference_progress = 0
        self.reference_initialized = True
        assert len(self.full_reference) == self.reference_len
        print("Loaded traj {} with {} elems".format(
            traj_fname, self.reference_len))

    def update_reference_progress(self, quad_position):
        reference_point = self.full_reference[self.reference_progress]
        reference_position_wf = np.array([reference_point.pose.position.x,
                                          reference_point.pose.position.y,
                                          reference_point.pose.position.z]).reshape((3, 1))
        distance = np.linalg.norm(reference_position_wf - quad_position)
        for k in range(self.reference_progress + 1, self.reference_len):
            reference_point = self.full_reference[k]
            reference_position_wf = np.array([reference_point.pose.position.x,
                                              reference_point.pose.position.y,
                                              reference_point.pose.position.z]).reshape((3, 1))
            next_point_distance = np.linalg.norm(reference_position_wf - quad_position)
            if next_point_distance > distance:
                break
            else:
                self.reference_progress = k
                distance = next_point_distance

    def callback_fly(self, data):
        assert NotImplementedError

    def experiment_report(self):
        print("experiment done")
        metrics_report = {}
        metrics_report['expert_usage'] = self.n_times_expert / \
                                         (self.n_times_net + self.n_times_expert) * 100.
        return metrics_report

    def callback_land(self, data):
        self.config.execute_nw_predictions = False

    @torch.no_grad()
    def callback_depth(self, data):
        if time.time() > self.plan_until_t:
            return
        if self.odometry is None:
            rospy.logwarn("No odom")
            return

        # t0 = time.time()
        # odom = self.odom_q.popleft()
        # while odom.header.stamp < data.header.stamp and self.odom_q:
        #     odom = self.odom_q.popleft()

        p = self.odometry.pose.pose.position
        q = self.odometry.pose.pose.orientation
        v = self.odometry.twist.twist.linear
        vz = v.z
        self.p = (p.x, p.y, p.z)
        self.q = (q.w, q.x, q.y, q.z)
        self.v = (v.x, v.y, v.z)

        depth = self.bridge.imgmsg_to_cv2(data)
        depth = np.float32(depth) / 1000
        depth[depth == 0] = 24.  # 24 meter barrier for uncertain pixels
        depth = 3 / np.clip(depth, 0.3, 24) - 0.6
        h, w = depth.shape
        _h = round((h - h * 0.82) / 2)
        _w = round((w - w * 0.82) / 2)
        depth = torch.as_tensor(depth[_h:-_h, _w:-_w])[None, None]
        depth = F.interpolate(depth, (36, 48), mode='nearest')
        depth = F.max_pool2d(depth, (3, 3))
        # depth_viz = np.clip(3 / (depth[0, 0].numpy() + 0.6) * 10, 0, 255)
        # cv2.imshow('depth', np.uint8(depth_viz))
        # cv2.waitKey(1)

        p, q, v = map(torch.as_tensor, (self.p, self.q, self.v))

        R = quaternion_to_matrix(q)

        env_R = R.clone()
        fwd = R[:, 0].clone()
        up = torch.zeros_like(fwd)
        fwd[2] = 0
        up[2] = 1
        fwd = fwd / torch.norm(fwd, 2, -1, keepdim=True)
        R = torch.stack([fwd, torch.cross(up, fwd), up], -1)

        if self.forward is None:
            self.forward = R[:, 0]

        assert self.p_target is not None
        target_v = torch.tensor([2000., 0, 0])
        target_v_norm = torch.norm(target_v, 2, -1, keepdim=True)
        max_speed = torch.tensor([self.config.target_speed])
        target_v = target_v / target_v_norm * max_speed

        state = [target_v[None] @ R, env_R[None, 2], self.margin[None]]
        global_v = v @ env_R.T
        if not self.config.no_odom:
            state.insert(0, global_v[None] @ R)
        state = torch.cat(state, -1)

        act, self.h = self.model(depth, state, self.h)
        a_pred, v_pred, *_ = (R @ act.reshape(3, -1)).unbind(-1)
        a_pred = a_pred - v_pred
        a_pred_debug = a_pred.tolist()
        a_pred[2] += 9.81

        thrust = torch.norm(a_pred)
        up_vec = a_pred / thrust
        self.forward = self.forward * 5 + target_v
        self.forward[2] = (self.forward[0] * up_vec[0] + self.forward[1] * up_vec[1]) / -up_vec[2]
        self.forward /= torch.norm(self.forward, 2, -1, True)
        left_vec = torch.cross(up_vec, self.forward)
        w, x, y, z = matrix_to_quaternion(torch.stack([
            self.forward, left_vec, up_vec
        ], 1)).tolist()

        self.cmd_pub.publish(
            control_mode=ControlCommand.ATTITUDE,
            armed=True,
            orientation=Quaternion(x, y, z, w),
            collective_thrust=thrust.item())
        self.a_set_pub.publish(*a_pred_debug)
        self.v_pub.publish(*global_v)
        # print("processing time:", time.time() - t0)

    def reset_queue(self):
        pass

    def callback_start(self, data):
        print("Callback START")
        self.pipeline_off = False

    def callback_off(self, data):
        print("Callback OFF")
        self.pipeline_off = True

    def maneuver_finished(self):
        return self.maneuver_complete

    def callback_gt_odometry(self, data):
        self.odometry = data
        # self.odom_q.append(data)

    def update_input_queues(self, data):
        if self.reference_initialized:
            quad_position = np.array([self.odometry.pose.pose.position.x,
                                        self.odometry.pose.pose.position.y,
                                        self.odometry.pose.pose.position.z]).reshape((3, 1))
            self.update_reference_progress(quad_position)
