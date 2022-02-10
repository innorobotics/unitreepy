# EXAMPLE HOW TO BUILD ROBOT FROM INTERFACE AND HANDLER

from unitreepy.base._handler import RobotHandler
from unitreepy.robots._build_robot import _build_robot
from unitreepy.legged_sdk import LowLevelInterface
from unitreepy.robots.a1.constants import POSITION_GAINS, DAMPING_GAINS, INIT_ANGLES, STAND_ANGLES
from types import SimpleNamespace
from numpy import array

CONSTANTS = SimpleNamespace()

CONSTANTS.POSITION_GAINS = POSITION_GAINS
CONSTANTS.DAMPING_GAINS = DAMPING_GAINS
CONSTANTS.INIT_ANGLES = INIT_ANGLES

# CREATE THE ROBOT 
interface = LowLevelInterface()
handler = RobotHandler(constants = CONSTANTS)
transmitter = interface.send
receiver = interface.receive

robot = _build_robot(handler, transmitter, receiver)


# start robot process and move to initial position
robot.start()
# move to desired angles
desired_angles = array(STAND_ANGLES)
robot.move_to(desired_angles)
# move to initial angles and stop robot process
robot.move_to_init()
robot.stop()


