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
        rclpy.init()
        ros_manager = RosTopicManager()
        self.ros_manager_thread = threading.Thread(target=rclpy.spin, args=(ros_manager,), daemon=True)
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
        for id in range(motor_num):
            cmd = MotorCmd()
            cmd.id = id
            data = MotorData()
            data.motorType = MotorType.GO_M8010_6
            cmd.motorType = MotorType.GO_M8010_6
            cmd.mode = queryMotorMode(MotorType.GO_M8010_6,MotorMode.FOC)
            self.motor_cmd_list.append(cmd)
            self.motor_data_list.append(data)         
    
    def locklegs(self, motor_pos=[0., 0., 0., 0., 0., 0.]):
        serial = SerialPort('/dev/ttyUSB0')
        for cmd, data, q in zip(self.motor_cmd_list, self.motor_data_list, motor_pos):
            cmd.q = q
            serial.sendRecv(cmd, data)

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