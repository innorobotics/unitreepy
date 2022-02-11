from unitreepy.base._handler import RobotHandler
from unitreepy.interfaces.gazebo_interface import GazeboInterface
from unitreepy.robots.a1.constants import STAND_ANGLES,INIT_ANGLES
import numpy as np
import time
import math

UPDATE_RATE = 1000

commsInterface = GazeboInterface(UPDATE_RATE)

robotHandler = RobotHandler(UPDATE_RATE)

def build_robot(handler, transmitter, receiver):
    handler.set_transmitter(transmitter)
    handler.set_receiver(receiver)
    robot = handler
    return robot

robot = build_robot(robotHandler,commsInterface.send, commsInterface.receive)

robot.start()

desired_angles = np.array(STAND_ANGLES)
robot.move_to(desired_angles)

initial_time = robot.state.time
current_time = 0

while robot.state.time - initial_time < 5:
    
    desired_angles = np.array(STAND_ANGLES)*(1 + 0.25 * math.sin(3*current_time))
    robot.set_angles(desired_angles)
    current_time = time.perf_counter()
    
robot.move_to_init()

robot.stop()
commsInterface.stop()