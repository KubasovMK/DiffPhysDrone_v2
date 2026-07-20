
from datetime import datetime
import json
import os
import shutil
import subprocess
import time

import rospy
from std_msgs.msg import Float32

from src.PlannerLearning import PlanLearning
from src.common import place_quad_at_start, setup_sim, MessageHandler
from src.ros_recorder import RosVideoRecorder

MAX_TIME_EXP = 100


import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--unity_start_pos', default=[[-20.,20.,0.,0]])
parser.add_argument('--length_straight', default=35.)
parser.add_argument('--max_rollouts', default=10)
parser.add_argument('--tree_spacings', default=[5])
parser.add_argument('--target_speed', default=13, type=float)
parser.add_argument('--track_global_traj', default=False)
parser.add_argument('--log_dir', default='logs')
parser.add_argument('--expert_folder', default="./data")
parser.add_argument('--pitch_angle', default=0)
parser.add_argument('--use_depth', default=True)
parser.add_argument('--use_rgb', default=False)
parser.add_argument('--quad_name', default='hummingbird')
parser.add_argument('--odometry_topic', default='/hummingbird/ground_truth/odometry')
parser.add_argument('--rgb_topic', default='/hummingbird/agile_autonomy/unity_rgb')
parser.add_argument('--depth_topic', default='/hummingbird/agile_autonomy/sgm_depth')
parser.add_argument('--input_update_freq', default=15)
parser.add_argument('--execute_nw_predictions', default=True)
parser.add_argument('--crashed_thr', default=0.18)
parser.add_argument('--verbose', default=False, action='store_true')
parser.add_argument('--net_weight', default='/mnt/Data/logs/drone_pos_runs_old/exps/single_no_odom_f/run1/checkpoint0004.pth')
parser.add_argument('--no_odom', default=False, action='store_true')
parser.add_argument('--margin', default=0.2, type=float)
args = parser.parse_args()
print(args)


class TestLoop:
    def __init__(self) -> None:
        self.settings = args
        self.msg_handler = MessageHandler()
        self.learner = PlanLearning(
            self.settings, mode="testing")
        tree_spacings = self.settings.tree_spacings
        removable_rollout_folders = os.listdir(self.settings.expert_folder)
        if len(removable_rollout_folders) > 0:
            removable_rollout_folders = [os.path.join(self.settings.expert_folder, d) \
                                            for d in removable_rollout_folders]
            removable_rollout_folders = [d for d in removable_rollout_folders if os.path.isdir(d)]
            for d in removable_rollout_folders:
                string = "rm -rf {}".format(d)
                os.system(string)
        self.fly_sub = rospy.Subscriber("/" + args.quad_name + "/agile_autonomy/compute_global_plan", Float32,
                                        self.callback_fly, queue_size=1)  # Receive and fly
        self.ctrl_rec = None
        self.recorder = None

        exp_name = datetime.now().strftime(f'%y%m%d_%H%M%S-{args.target_speed}')
        self.exp_log_dir = os.path.join(self.settings.log_dir, exp_name)
        os.makedirs(self.exp_log_dir)
        os.system(f'cp -r src *.py {self.exp_log_dir}')
        shutil.copy(args.net_weight, self.exp_log_dir)
        self.rollout_idx = 0

    def start_experiment(self):
        self.msg_handler.publish_reset()
        place_quad_at_start(self.msg_handler)
        print("Doing experiment {}".format(self.rollout_idx))
        self.msg_handler.publish_save_pc()
    
    def callback_fly(self, data):
        self.start_recording()
        rospy.sleep(1)
        self.learner.callback_fly(data)

    def start_recording(self):
        self.stop_recording()
        self.ctrl_rec = subprocess.Popen([
            'rosbag', 'record', '-j', '-O', f'{self.exp_log_dir}/{self.rollout_idx}.bag',
            # args.depth_topic,
            '/hummingbird/autopilot/control_command_input',
            '/hummingbird/ground_truth/odometry',
            '/hummingbird/control_command',
            '/e2e_planner_v2/a_set',
            '/e2e_planner_v2/v',
            '/e2e_planner_v2/closest_distance',
        ])
        self.recorder = RosVideoRecorder(
            args.rgb_topic, f'{self.exp_log_dir}/{self.rollout_idx}.mp4')

    def stop_recording(self):
        if self.recorder is not None:
            self.recorder.close()
            self.recorder = None
        if self.ctrl_rec is not None:
            self.ctrl_rec.send_signal(2)  # SIGINT
            self.ctrl_rec.wait()
            self.ctrl_rec = None

    def test(self, spacing):
        self.msg_handler.publish_tree_spacing(spacing)
        self.msg_handler.publish_obj_spacing(spacing)
        output_file_buffer = os.path.join(self.exp_log_dir, "log.json")
        with open(output_file_buffer, 'a') as f:
            json.dump(args.__dict__, f)
            f.write('\n')
        # Start Experiment
        while self.rollout_idx < self.settings.max_rollouts and not rospy.is_shutdown():
            self.learner.maneuver_complete = False  # Just to be sure
            setup_sim(self.msg_handler, config=self.settings)
            self.start_experiment()

            start = time.time()
            self.expert_done = False  # Re-init to be sure
            while not self.learner.maneuver_complete and not self.learner.crashed and not rospy.is_shutdown():
                rospy.sleep(0.1)
                duration = time.time() - start
                if duration > MAX_TIME_EXP:
                    print("timeout.")
                    break
            if not self.learner.maneuver_complete:
                self.learner.publish_stop_recording_msg()
            if self.learner.planner_succed:
                # final logging
                metrics_experiment = self.learner.experiment_report()
                print("------- {} Rollout ------------".format(self.rollout_idx))
                for name, value in metrics_experiment.items():
                    print("{} is {:.3f}".format(name, value))
                print("-------------------------------")
                self.rollout_idx += 1
                rollout_dir = os.path.join(self.settings.expert_folder,
                                                sorted(os.listdir(self.settings.expert_folder))[-1])
                # Wait one second to stop recording
                time.sleep(1)
                if self.settings.verbose:
                    # Mv data record to log folder
                    move_string = "mv {} {}".format(
                        rollout_dir, self.exp_log_dir)
                    os.system(move_string)
                else:
                    print("Rollout dir is {}".format(rollout_dir))
                    shutil.rmtree(rollout_dir)
                # Save latest version of report buffer
                with open(output_file_buffer, 'a') as f:
                    json.dump(metrics_experiment, f)
                    f.write('\n')
            else:
                # Wait one second to stop recording
                time.sleep(1)
                # remove folder
                rollout_dir = os.path.join(self.settings.expert_folder,
                                            sorted(os.listdir(self.settings.expert_folder))[-1])
                rm_string = "rm -rf {}".format(rollout_dir)
                os.system(rm_string)
            self.stop_recording()


if __name__ == '__main__':
    rospy.init_node('e2e_planner_v2')
    try:
        test_loop = TestLoop()
        for tree_spacing in args.tree_spacings:
            test_loop.test(spacing=tree_spacing)
    except rospy.ROSInterruptException:
        exit()
