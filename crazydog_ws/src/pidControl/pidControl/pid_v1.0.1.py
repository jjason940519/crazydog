import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
from pid_example import PID
import threading

import time
import sys
sys.path.append('/home/crazydog/crazydog/crazydog_ws/src/pidControl/pidControl/modules/unitree_actuator_sdk/lib')
from crazydog_ws.src.pidControl.pidControl.modules.unitree_actuator_sdk import *
import traceback
import math

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
        self.imu_subscriber = self.create_subscription(Float32MultiArray,'imu', self.imu_callback, 1)
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

    def imu_callback(self, msg):
        qua_x = msg.orientation.x
        qua_y = msg.orientation.y
        qua_z = msg.orientation.z
        qua_w = msg.orientation.w
        self.angularvelocity_y = msg.angular_velocity.y
        self.pitch_init = (
            math.asin(2 * (qua_w * qua_y - qua_z * qua_x)))
        
        t3 = 2 * (qua_w * qua_z + qua_x * qua_y)
        t4 = 1 - 2 * (qua_y * qua_y + qua_z * qua_z)
        self.yaw_init = math.atan2(t3, t4)

    def getImuOrientation(self) -> float:
        return -self.pitch_init, self.yaw_init
    
    def send_foc_command(self, current_right, current_left):
        msg = Float32MultiArray()
        # msg.data = 'Hello World: %d' % self.i
        msg.data = [current_right, current_left]
        self.foc_command_publisher.publish(msg)

    def get_foc_status(self):
        return self.foc_right, self.foc_left

class robotController():
    def __init__(self) -> None:
        rclpy.init()
        self.ros_manager = RosTopicManager()
        self.ros_manager_thread = threading.Thread(target=rclpy.spin, args=(self.ros_manager,), daemon=True)
        self.ros_manager_thread.start()
        self.running_flag = False

        self.motor_cmd_list = []
        self.motor_data_list = []

        self.angular_pid = PID(10, 0, 1)
        # self.balance_pid.output_limits = (-10, 10)
        self.velocity_pid = PID(0.008, 0.0, 0.00001)
        # self.velocity_pid.output_limits = (0, 0)
        self.angularvelocity_pid = PID(65, 0, 0)

        self.yaw_pid = PID(350, 0, 0.4)

        self.init_unitree_motor()
        self.locklegs()
        self.startController()

    def init_unitree_motor(self, motor_num=6):
        pass       
    
    def locklegs(self, motor_pos=[0., 0., 0., 0., 0., 0.]):
        pass

    def controller(self):
        while self.running_flag:

            pass

    def startController(self):
        self.prev_pitch = 0
        self.pid_thread = threading.Thread(target=self.controller)
        self.running_flag = True
        self.pid_thread.start()

    def disableController(self):
        self.running_flag = False
        if self.pid_thread is not None:
            self.pid_thread.join()
        print("disable controller")

def main(args=None):
    robot = robotController()
    command_dict = {
        "d": robot.disableController,
        "start": robot.startController,
        # "get": robot_motor.getControllerPIDParam,
        # "clear": robot_motor.mc.cleanerror,
    }

    while True:
        try:
            cmd = input("CMD :")
            if cmd in command_dict:
                command_dict[cmd]()
            elif cmd == "exit":
                robot.disableController()
                break

        except Exception as e:
            traceback.print_exc()
            break


if __name__ == '__main__':
    main()