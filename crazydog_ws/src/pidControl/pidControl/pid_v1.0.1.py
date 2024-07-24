import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from crazydog_ws.src.pidControl.pidControl.pid_example import PID
import threading


class focMotor():
    def __init__(self):
        self.angle = 0.0    # rad
        self.speed = 0.0    # rpm
        self.current = 0.0  
        self.temperature = 0.0  # degree C

class RosTopicManager(Node):
    def __init__(self):
        super().__init__('ros_topic_manager')
        self.foc_data_subscriber = self.create_subscription(Float32MultiArray,'foc_msg', self.foc_callback, 1)
        # self.imu_subscriber = self.create_subscription(Float32MultiArray,'imu', self.imu_callback, 1)
        self.foc_command_publisher = self.create_publisher(Float32MultiArray, 'foc_command', 1)
        self.foc_right = focMotor()
        self.foc_left = focMotor()

    def foc_callback(self, msg):
        if msg.data[0] == 513:
            self.foc_right.angle = msg[1]
            self.foc_right.speed = msg[2]
            self.foc_right.current = msg[3]
            self.foc_right.temperature = msg[4]
        elif msg.data[0] == 514:
            self.foc_left.angle = msg[1]
            self.foc_left.speed = msg[2]
            self.foc_left.current = msg[3]
            self.foc_left.temperature = msg[4]
    
    def send_foc_command(self, current_right, current_left):
        msg = Float32MultiArray()
        # msg.data = 'Hello World: %d' % self.i
        msg.data = [current_right, current_left]
        self.foc_command_publisher.publish(msg)

    def get_foc_status(self):
        return self.foc_right, self.foc_left

class robotController():
    def __init__(self) -> None:
        ros_manager = RosTopicManager()
        subscriber_thread = threading.Thread(target=rclpy.spin, args=(ros_manager,), daemon=True)
        subscriber_thread.start()
        # lock_legs()


def main(args=None):
    rclpy.init(args=args)

    robot_control = RosTopicManager()

    rclpy.spin(robot_control)

    robot_control.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()