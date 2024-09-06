import time
import math
import sys
sys.path.append('/home/crazydog/crazydog/crazydog_ws/src/lqr_control/lqr_control/utils')
import unitree_motor_command_thread as um
import threading


unitree = um.unitree_communication('/dev/unitree-l')
MOTOR1 = unitree.createMotor(motor_number = 1,initalposition = 0.669,MAX=8.475,MIN=-5.364)
MOTOR2 = unitree.createMotor(motor_number = 2,initalposition = 3.815,MAX=26.801,MIN=-1)
unitree2 = um.unitree_communication('/dev/unitree-r')
MOTOR4 = unitree2.createMotor(motor_number = 4,initalposition = 1.247+2*math.pi,MAX=5.364,MIN=-8.475)
MOTOR5 = unitree2.createMotor(motor_number = 5,initalposition = 2.970,MAX=1,MIN=-26.801)


def disableUnitreeMotor():

    unitree.disableallmotor()
    unitree2.disableallmotor()

def init_unitree_motor():
    if MOTOR1.data.q >= MOTOR1.inital_position and MOTOR4.data.q <= MOTOR4.inital_position:
        while MOTOR1.data.q >= MOTOR1.inital_position and MOTOR4.data.q <= MOTOR4.inital_position:
            unitree.position_force_velocity_cmd(motor_number = 1,kp = 2,kd = 0.02, position = MOTOR1.data.q - 0.1 ,torque = 0, velocity = 0)
            unitree2.position_force_velocity_cmd(motor_number = 4 ,kp = 2,kd = 0.02, position = MOTOR4.data.q + 0.1 ,torque = 0, velocity= 0)
            time.sleep(0.01)
        # for i in range(8):
        #     unitree.position_force_velocity_cmd(motor_number = 1,kp = i,kd = 0.1, position = MOTOR1.data.q)
        #     unitree2.position_force_velocity_cmd(motor_number = 4 ,kp = i,kd = 0.1, position = MOTOR4.data.q)
        time.sleep(0.1)
        unitree.inital_check()
        unitree2.inital_check()
    else:
        print("inital fail")

def locklegs():

    while MOTOR1.data.q >= MOTOR1.inital_position + 0.33*6.33 and MOTOR4.data.q  <= MOTOR4.inital_position  :            
        unitree.position_force_velocity_cmd(motor_number = 1,kp = 2,kd = 0.02, position = MOTOR1.data.q + 0.1 , velocity = 0)
        unitree2.position_force_velocity_cmd(motor_number = 4 ,kp = 2,kd = 0.02, position = MOTOR4.data.q - 0.1 , velocity= 0)
        time.sleep(0.01)
    for i in range(36):                        
        unitree.position_force_velocity_cmd(motor_number = 1,kp = i,kd = 0.12, position = MOTOR1.inital_position + 0.33*6.33)
        unitree2.position_force_velocity_cmd(motor_number = 4 ,kp = i,kd = 0.12, position = MOTOR4.inital_position - 0.33*6.33)
        time.sleep(0.01)
    while MOTOR2.data.q >= MOTOR2.inital_position + 0.33*6.33*1.6 and MOTOR5.data.q  <= MOTOR5.inital_position - 0.33*6.33*1.6:
        unitree.position_force_velocity_cmd(motor_number = 2,kp = 0,kd = 0.16, position = 0 ,torque = 0, velocity = 0.01)
        unitree2.position_force_velocity_cmd(motor_number = 5 ,kp = 0,kd = 0.16, position = 0 ,torque = 0, velocity=-0.01)
        time.sleep(0.01)
    for i in range(36):                        
        unitree.position_force_velocity_cmd(motor_number = 2,kp = i,kd = 0.15, position = MOTOR2.inital_position + 0.6*6.33*1.6)
        unitree2.position_force_velocity_cmd(motor_number = 5 ,kp = i,kd = 0.15, position = MOTOR5.inital_position - 0.6*6.33*1.6)
        time.sleep(0.1)

def enable():

    unitree.enableallmotor()
    unitree2.enableallmotor()

def main():

    command_dict = {
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
                disableUnitreeMotor()
                break
        except KeyboardInterrupt:
            disableUnitreeMotor()
            break

if __name__ == '__main__':
    main()