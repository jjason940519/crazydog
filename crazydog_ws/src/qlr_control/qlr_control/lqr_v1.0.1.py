import rclpy
from rclpy.node import Node

from std_msgs.msg import Float32MultiArray
import threading

import time
import sys
# sys.path.append('/home/crazydog/crazydog/crazydog_ws/src/pidControl/pidControl/modules/unitree_actuator_sdk/lib')
# from crazydog_ws.src.pidControl.pidControl.modules.unitree_actuator_sdk import *
import traceback
import math
import numpy as np
from LQR_pin import InvertedPendulumLQR

WHEEL_RADIUS = 0.08     # m
WHEEL_MASS = 0.695  # kg
URDF_PATH = "/home/crazydog/crazydog/crazydog_ws/src/qlr_control/qlr_control/robot_models/big bipedal robot v1/urdf/big bipedal robot v1.urdf"

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
        self.imu_subscriber = self.create_subscription(Float32MultiArray,'hansfree/imu', self.imu_callback, 1)
        self.foc_command_publisher = self.create_publisher(Float32MultiArray, 'foc_command', 1)
        self.foc_right = focMotor()
        self.foc_left = focMotor()
        # self.pitch_last = 0
        # self.pitch_dot = 0
        self.row = 0
        self.row_dot = 0
        self.dt = 1/300


    def foc_callback(self, msg):
        if msg.data[0] == 514.: # motor right
            self.foc_right.angle = -msg.data[1]
            self.foc_right.speed = -msg.data[2]
            self.foc_right.current = -msg.data[3]
            self.foc_right.temperature = msg.data[4]
        elif msg.data[0] == 513.:   # motor left
            self.foc_left.angle = msg.data[1]
            self.foc_left.speed = msg.data[2]
            self.foc_left.current = msg.data[3]
            self.foc_left.temperature = msg.data[4]
    
    def send_foc_command(self, current_right, current_left):
        msg = Float32MultiArray()
        torque_const = 0.3  # N-m/A
        msg.data = [current_right/torque_const, current_left/torque_const]
        self.foc_command_publisher.publish(msg)

    def get_foc_status(self):
        return self.foc_left, self.foc_right
    
    def imu_callback(self, msg):
        qua_x = msg.orientation.x
        qua_y = msg.orientation.y
        qua_z = msg.orientation.z
        qua_w = msg.orientation.w
        # *(180/math.pi)+1.5
        t0 = +2.0 * (qua_w * qua_x + qua_y * qua_z)
        t1 = +1.0 - 2.0 * (qua_x * qua_x + qua_y * qua_y)
        self.row = math.atan2(t0, t1)
        # self.pitch = -(math.asin(2 * (qua_w * qua_y - qua_z * qua_x)) - self.pitch_bias) 
        self.row_dot = msg.angular_velocity.x
        # self.pitch_dot = (self.pitch - self.pitch_last) / self.dt
        # self.pitch_last = self.pitch
        # self.pitch_dot = msg.angular_velocity.y

    def get_orientation(self):
        return self.row, self.row_dot

class robotController():
    def __init__(self) -> None:
        rclpy.init()
        Q = np.diag([0, 1.0, 1.0, 1.0])
        R = np.diag(np.diag([1e-6]))
        q = np.array([0., 0., 0., 0., 0., 0., 1.,
                            0., -1.18, 2.0, 1., 0.,
                            0., -1.18, 2.0, 1., 0.])
        self.lqr_controller = InvertedPendulumLQR(pos=q, 
                                                  urdf=URDF_PATH, 
                                                  wheel_r=WHEEL_RADIUS, 
                                                  M=WHEEL_MASS, Q=Q, R=R, 
                                                  delta_t=1/200, 
                                                  show_animation=False)
        self.ros_manager = RosTopicManager()
        self.ros_manager_thread = threading.Thread(target=rclpy.spin, args=(self.ros_manager,), daemon=True)
        self.ros_manager_thread.start()
        self.running_flag = False

        self.motor_cmd_list = []
        self.motor_data_list = []
        

        self.init_unitree_motor()
        self.locklegs()

    def init_unitree_motor(self, motor_num=6):
        pass       
    
    def locklegs(self, motor_pos=[0., 0., 0., 0., 0., 0.]):
        pass

    def startController(self):
        self.prev_pitch = 0
        self.lqr_thread = threading.Thread(target=self.controller)
        self.running_flag = True
        self.lqr_thread.start()

    def controller(self):
        print('controller start')
        X = np.zeros((4, 1))    # X = [x, x_dot, theta, theta_dot]
        U = np.zeros((1, 1))
        t0 = time.time()
        count = 0
        count_drop = 0
        while self.running_flag:
            count += 1
            t1 = time.time()
            dt = t1 - t0
            t0 = t1

            X_last = np.copy(X)
            foc_status_left, foc_status_right = self.ros_manager.get_foc_status()
            # get motor data
            X[1, 0] = (foc_status_left.speed + foc_status_right.speed) / 2
            X[0, 0] = X_last[0, 0] + X[1, 0] * dt * WHEEL_RADIUS     # wheel radius 0.08m
            # get IMU data
            X[2, 0], X[3, 0] = self.ros_manager.get_orientation()
            # if abs(X[2, 0]) > math.radians(50):     # constrain
            #     u[0, 0] = 0.0
            #     self.ros_manager.send_foc_command(u[0,0], u[0,0])
            #     continue

            # get u from lqr
            U = np.copy(self.lqr_controller.lqr_control(X))
            self.ros_manager.send_foc_command(U[0, 0], U[0, 0])


    def disableController(self):
        self.running_flag = False
        if self.lqr_thread is not None:
            self.lqr_thread.join()
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