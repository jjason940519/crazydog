import rclpy
import threading
import time
import math
import numpy as np
from utils.LQR_pin import InvertedPendulumLQR
from utils.pid import PID
from utils.ros_manager import RosTopicManager
from utils.motor_getready import disableUnitreeMotor, init_unitree_motor, locklegs, enable

import unitree_motor_command as um

WHEEL_RADIUS = 0.08     # m
WHEEL_MASS = 0.695  # kg
WHEEL_DISTANCE = 0.355
URDF_PATH = "/home/crazydog/crazydog/crazydog_ws/src/lqr_control/lqr_control/robot_models/big bipedal robot v1/urdf/big bipedal robot v1.urdf"
MID_ANGLE = 0.05
TORQUE_CONSTRAIN = 1.5

class robotController():
    def __init__(self) -> None:
        rclpy.init()
        # K: [[ 2.97946709e-07  7.36131891e-05 -1.28508761e+01 -4.14185118e-01]]
        Q = np.diag([1e-9, 0.1, 1.0, 1e-4])       # 1e-9, 1e-9, 0.01, 1e-6; 1e-9, 0.1, 1.0, 1e-4
        R = np.diag(np.diag([2e-2]))
        q = np.array([0., 0., 0., 0., 0., 0., 1.,
                            0., -1.18, 2.0, 1., 0.,
                            0., -1.18, 2.0, 1., 0.])
        self.lqr_controller = InvertedPendulumLQR(pos=q, 
                                                  urdf=URDF_PATH, 
                                                  wheel_r=WHEEL_RADIUS, 
                                                  M=WHEEL_MASS, Q=Q, R=R, 
                                                  delta_t=1/300, 
                                                  show_animation=False)
        self.ros_manager = RosTopicManager()
        self.ros_manager_thread = threading.Thread(target=rclpy.spin, args=(self.ros_manager,), daemon=True)
        self.ros_manager_thread.start()
        self.running_flag = False
        self.turning_pid = PID(0.5, 0, 0.0001)

    def startController(self):
        self.prev_pitch = 0
        self.lqr_thread = threading.Thread(target=self.controller)
        self.running_flag = True
        self.lqr_thread.start()

    def get_yaw_speed(self, speed_left, speed_right):
        delta_speed = (speed_left-speed_right)* (2*np.pi*WHEEL_RADIUS)/60
        yaw_speed = delta_speed / WHEEL_DISTANCE
        return yaw_speed

    def controller(self):
        self.ros_manager.get_logger().info('controller start')
        X = np.zeros((4, 1))    # X = [x, x_dot, theta, theta_dot]
        U = np.zeros((1, 1))
        t0 = time.time()
        X_ref = np.zeros((4, 1))
        yaw_ref = 0.
        yaw_speed = 0.
        start_time = time.time()

        while self.running_flag:
            with self.ros_manager.ctrl_condition:
                self.ros_manager.ctrl_condition.wait()
                # print(self.ros_manager.imu_update)
                # self.ros_manager.ctrl_update = False
                # print('imu false')
                t1 = time.time()
                dt = t1 - t0
                # print('freq:', 1/dt)
                t0 = t1
                X_ref[2, 0], yaw_ref = self.ros_manager.get_joy_vel()
                X_ref[2, 0] += MID_ANGLE
                X_last = np.copy(X)
                foc_status_left, foc_status_right = self.ros_manager.get_foc_status()
                # get motor data
                X[1, 0] = (foc_status_left.speed + foc_status_right.speed) / 2 * (2*np.pi*WHEEL_RADIUS)/60
                X[0, 0] = X_last[0, 0] + X[1, 0] * dt * WHEEL_RADIUS     # wheel radius 0.08m
                # get IMU data
                X[2, 0], X[3, 0] = self.ros_manager.get_orientation()
                # X[2, 0], _ = self.ros_manager.get_orientation()
                # X[3, 0] = (X[2, 0] - X_last[2, 0]) / dt
                if abs(X[2, 0]) > math.radians(25):     # constrain
                    # U[0, 0] = 0.0
                    self.ros_manager.send_foc_command(0.0, 0.0)
                    continue
                
                # get u from lqr
                U = np.copy(self.lqr_controller.lqr_control(X, X_ref))
                # print(X)
                # print('u:', U[0, 0])
                
                # time.sleep(3e-3)
                # print(X)
                # print(X_desire[2,0])
                yaw_speed = self.get_yaw_speed(foc_status_left.speed, foc_status_right.speed)
                yaw_torque = self.turning_pid.update(yaw_ref, yaw_speed, dt)

                if (time.time()-start_time) < 3:
                    yaw_torque = 0.0
                motor_command_left = U[0, 0] + yaw_torque
                motor_command_right = U[0, 0] - yaw_torque
                # print(yaw_torque)

                motor_command_left = max(-1.5, min(motor_command_left, 1.5))
                motor_command_right = max(-1.5, min(motor_command_right, 1.5))
                # print(motor_command_left, motor_command_right)

                self.ros_manager.send_foc_command(motor_command_left, motor_command_right)


    def disableController(self):
        self.running_flag = False
        self.ros_manager.send_foc_command(0.0, 0.0)
        if self.lqr_thread is not None:
            self.lqr_thread.join()
        self.ros_manager.get_logger().info("disable controller")
    



def main(args=None):
    robot = robotController()
    command_dict = {
        "start": robot.startController,
        "d": disableUnitreeMotor,
        "i": init_unitree_motor,
        "l": locklegs,
        "s": enable,
    }

    while True:
        try:
            cmd = input("CMD :")
            if cmd in command_dict:
                command_dict[cmd]()
            elif cmd == "exit":
                robot.disableController()
                break
        except KeyboardInterrupt:
            robot.disableController()
            break
        # except Exception as e:
        #     traceback.print_exc()
        #     break


if __name__ == '__main__':
    main()