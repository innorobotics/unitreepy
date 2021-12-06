def A1Robot(real=True, highLevel=False,publishXPP = True):
    if real:
        from legged_sdk import LowLevelInterface,HighLevelInterface
        from types import SimpleNamespace
        from pyunitree.base._handler import RobotHandler
        from pyunitree.robots.a1.constants import POSITION_GAINS, DAMPING_GAINS, INIT_ANGLES


        CONSTANTS = SimpleNamespace()
        CONSTANTS.POSITION_GAINS = POSITION_GAINS
        CONSTANTS.DAMPING_GAINS = DAMPING_GAINS
        CONSTANTS.INIT_ANGLES = INIT_ANGLES

        interface = LowLevelInterface()
        
        robot = RobotHandler(constants = CONSTANTS)
        robot.transmitter = interface.send
        robot.receiver = interface.receive

        if highLevel:
            robot.highInterface = HighLevelInterface()   

    else:
        from pyunitree.interfaces.gazeboInterface import GazeboInterface

        robot = GazeboInterface(publishXPP=publishXPP)
        robot.slowDownSim()
        
    return robot