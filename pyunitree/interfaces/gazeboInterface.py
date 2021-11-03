from multiprocessing.sharedctypes import RawValue
from pyunitree.parsers.gazebo import GazeboMsgParser
from pyunitree.robots.a1.constants import INIT_ANGLES,POSITION_GAINS,DAMPING_GAINS
from pyunitree.utils._pos_profiles import p2p_cos_profile

import rospy
from unitree_legged_msgs.msg import MotorCmd,MotorState
from sensor_msgs.msg import Imu
from geometry_msgs.msg import WrenchStamped

from logging import info
import time
import numpy as np

from pyunitree.base.daemon import Daemon
from multiprocessing import Manager
    
class GazeboInterface(Daemon):
    controllerNames  =  [
                        "/a1_gazebo/FR_hip_controller",
                        "/a1_gazebo/FR_thigh_controller",
                        "/a1_gazebo/FR_calf_controller",
                        "/a1_gazebo/FL_hip_controller",
                        "/a1_gazebo/FL_thigh_controller",
                        "/a1_gazebo/FL_calf_controller",
                        "/a1_gazebo/RR_hip_controller",
                        "/a1_gazebo/RR_thigh_controller",
                        "/a1_gazebo/RR_calf_controller",
                        "/a1_gazebo/RL_hip_controller",
                        "/a1_gazebo/RL_thigh_controller",
                        "/a1_gazebo/RL_calf_controller"
                        ]

    footContactNames = [ 
                        "/visual/FR_foot_contact/the_force",
                        "/visual/FL_foot_contact/the_force",
                        "/visual/RR_foot_contact/the_force",
                        "/visual/RL_foot_contact/the_force"
                        ]

    #Joint names from JointState message
    motorNames       = [
                        "FR_hip_joint",
                        "FR_thigh_joint",
                        "FR_calf_joint",
                        "FL_hip_joint",
                        "FL_thigh_joint",
                        "FL_calf_joint",
                        "RR_hip_joint",
                        "RR_thigh_joint",
                        "RR_calf_joint",
                        "RL_hip_joint",
                        "RL_thigh_joint",
                        "RL_calf_joint"
                        ]

    #Maps joint name to it's location in position and velocity idx in joint state
    motorMappings = {name:i for i,name in enumerate(motorNames)}

    def __init__(self,updateRate = -1,name = "Gazebo state listener"):
        super(GazeboInterface,self).__init__(updateRate,name)
        
        self.parser = GazeboMsgParser()
        self.jointNames = None
        self.position = [0]*12
        self.velocity = [0]*12
        self.footForces = [[0]*3]*4
        self.imu = [0]*10
        self.time = 0
        self.initTime = 0

        self.__shared = Manager().Namespace()
        self.__shared.joint_angles = [0]*12
        self.__shared.cmd = [0]*60

        self.sharedStateSize = 39
        self.sharedStateType = np.float32

        data = np.zeros(self.sharedStateSize, dtype=self.sharedStateType)

        self.stateIsValid = RawValue("b",False)
        self.rawRemotePtr = None
        self.initSharedStateArray(39,"RobotState")

    def processInit(self):
        self.node = rospy.init_node('unitreepy_node')
    
        self.imuSub = rospy.Subscriber("/trunk_imu", Imu, self.imuVectorCallback)

        self.footForceSubs = [rospy.Subscriber(name, WrenchStamped, self.footForceVectorCallback,(idx)) 
                                                            for idx,name in enumerate(GazeboInterface.footContactNames)]

        self.motorSubs = [rospy.Subscriber(name+"/state", MotorState, self.motorVectorCallback,(idx)) 
                                                            for idx,name in enumerate(GazeboInterface.controllerNames)]

        self.servoPublishers = [rospy.Publisher(controllerName+"/command", MotorCmd) 
                                                            for controllerName in GazeboInterface.controllerNames]

        self.initTime = rospy.get_time()
        info("Unitreepy Gazebo listener: Attempting to receive initial position from Gazebo")
        while not self.stateIsValid.value:
            time.sleep(0.001)
        self.moveStateToShared()
        info("Unitreepy Gazebo listener: Initial state received")

    def action(self):
        try:
            command = self.__shared.cmd
            self.time = rospy.get_time()-self.initTime
            self.moveStateToShared()
            self.sendCommand(command)
            return True
        except BrokenPipeError:
            return False
    
    def onStart(self):
        while not self.stateIsValid.value:
            time.sleep(0.01)
        
    #Utils
    
    def sendCommand(self,command):
        motorCmd = MotorCmd()
        for motorId in range(12):
            motorCmd.mode = 0x0A
            motorCmd.q=command[motorId * 5]
            motorCmd.Kp=command[motorId * 5+1]
            motorCmd.dq=command[motorId * 5+2]
            motorCmd.Kd=command[motorId * 5+3]
            motorCmd.tau=command[motorId * 5+4]
            self.servoPublishers[motorId].publish(motorCmd)

    def buildCommand(self,
                      desired_pos=np.zeros(12),
                      desired_vel=np.zeros(12),
                      desired_torque=np.zeros(12),
                      position_gains=np.zeros(12),
                      damping_gains=np.zeros(12)):

        command = np.zeros(60)

        for motor_id in range(12):
            command[motor_id * 5] = desired_pos[motor_id]
            command[motor_id * 5 + 1] = position_gains[motor_id]
            command[motor_id * 5 + 2] = desired_vel[motor_id]
            command[motor_id * 5 + 3] = damping_gains[motor_id]
            command[motor_id * 5 + 4] = desired_torque[motor_id]

        return command

    def moveStateToShared(self):
        self.__shared.joint_angles = np.array(self.position)

        footforce = np.array([force[2] for force in self.footForces])

        compressedState = np.hstack([self.imu,footforce,self.position,self.velocity,[self.time]])

        # NOTE Buffer manipulation
        np.copyto(self.rawStateBuffer, compressedState)
        
    def motorVectorCallback(self,msg,idx):
        self.position[idx] = msg.q
        self.velocity[idx] = msg.dq

    def imuVectorCallback(self,msg):
        self.imu = self.parser.vectorizeImuMsg(msg)
        self.stateIsValid.value = 1

    def footForceVectorCallback(self,msg,footIdx):
        self.footForces[footIdx] = self.parser.vectorizeEeForce(msg)

    def send(self,cmd):
        self.__shared.cmd = cmd
    
    def set_command(self,cmd):
        self.__shared.cmd = cmd
        
    def set_torques(self,desiredTorques):
        self.send(self.buildCommand(desired_torque=desiredTorques))

    def set_angles(self,desiredPos):
        self.send(self.buildCommand(desired_pos=desiredPos,position_gains=POSITION_GAINS,damping_gains=DAMPING_GAINS))

    def move_to(self, desired_position, terminal_time=3):
        """Move to desired position with poitn to point """
        initial_position = self.__shared.joint_angles
        init_time = time.perf_counter()
        actual_time = 0

        while actual_time <= terminal_time:
            actual_time = time.perf_counter() - init_time
            position, _ = p2p_cos_profile(actual_time,
                                          initial_pose=initial_position,
                                          final_pose=desired_position,
                                          terminal_time=terminal_time)
            self.set_angles(position)
                
    def move_to_init(self):
        self.move_to(np.array(INIT_ANGLES))
