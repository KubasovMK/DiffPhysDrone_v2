import time
import rospy

from sensor_msgs.msg import Imu
from quadrotor_msgs.msg import ControlCommand
from geometry_msgs.msg import Quaternion

rospy.init_node("abl")

az = 0
def imu_callback(data: Imu):
    global az
    az = data.linear_acceleration.z

rospy.Subscriber("/hummingbird/ground_truth/imu", Imu, imu_callback, queue_size=1)

cmd_pub = rospy.Publisher("/hummingbird/autopilot/control_command_input", ControlCommand, queue_size=1)
y = []
for x in range(1, 28):
    cmd_pub.publish(
        control_mode=ControlCommand.ATTITUDE,
        armed=True,
        orientation=Quaternion(0, 0, 0, 1),
        collective_thrust=x)
    time.sleep(0.2)
    y.append(az)

print(y)
