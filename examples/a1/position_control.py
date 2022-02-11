from unitreepy.robots.a1.constants import POSITION_GAINS, DAMPING_GAINS, INIT_ANGLES, STAND_ANGLES
from unitreepy.base._handler import RobotHandler
from unitreepy.interfaces.shared_state import SHM_IMPORTED
from legged_sdk import LowLevelInterface
from types import SimpleNamespace
from numpy import array, sin

CONSTANTS = SimpleNamespace()
CONSTANTS.POSITION_GAINS = POSITION_GAINS
CONSTANTS.DAMPING_GAINS = DAMPING_GAINS
CONSTANTS.INIT_ANGLES = INIT_ANGLES


interface = LowLevelInterface()

robot = RobotHandler(constants=CONSTANTS)
robot.transmitter = interface.send
robot.receiver = interface.receive

robot.start()
desired_angles = array(STAND_ANGLES)
robot.move_to(desired_angles)
initial_time = robot.state.time

time = 0
while time < 5:
    time = robot.state.time - initial_time
    desired_angles = array(STAND_ANGLES)*(1 + 0.25 * sin(3*time))

    robot.set_angles(desired_angles)

robot.move_to_init()
robot.stop()
