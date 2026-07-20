import rospy
from subprocess import Popen, PIPE
from sensor_msgs.msg import Image
from cv_bridge import CvBridge


class RosVideoRecorder:
    def __init__(self, topic, output, fps=15, stream=False) -> None:
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber(topic, Image, self.callback_image, queue_size=1)
        self.stream = stream
        self.p = None
        self.fps = fps
        self.output = output

    def callback_image(self, data):
        image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        h, w, _ = image.shape
        if self.p is None:
            command = [
                "/usr/bin/ffmpeg",
                '-y',  # overwrite output file if it exists
                '-f', 'rawvideo',
                '-vcodec','rawvideo',
                '-s', f'{w}x{h}',  # size of one frame
                '-pix_fmt', 'bgr24',
                '-r', f'{self.fps}',  # frames per second
                '-i', '-',  # The imput comes from a pipe
                '-s', f'{w//2*2}x{h//2*2}',
                '-an',  # Tells FFMPEG not to expect any audio
                '-loglevel', 'error',
                '-c:v', 'h264_nvenc',
                # '-pix_fmt', 'yuv420p'
            ]
            self.p = Popen(command + [self.output], stdin=PIPE)
        self.p.stdin.write(image.tobytes())
    
    def close(self):
        self.image_sub.unregister()
        if self.p is not None:
            self.p.stdin.close()
            self.p.wait()


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('topic')
    parser.add_argument('output')
    parser.add_argument('--fps', default=15)
    parser.add_argument('--stream', default=False, action='store_true')
    args = parser.parse_args()
    rospy.init_node('recorder')
    r = RosVideoRecorder(args.topic, args.output, fps=args.fps, stream=args.stream)
    rospy.spin(20)
    r.close()
    # rospy.spin()
