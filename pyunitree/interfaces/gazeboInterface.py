import logging
from multiprocessing.sharedctypes import RawValue
from pyunitree.parsers.gazebo import GazeboMsgParser
from pyunitree.robots.a1.constants import INIT_ANGLES,POSITION_GAINS,DAMPING_GAINS
from pyunitree.utils._pos_profiles import p2p_cos_profile
from gazebo_msgs.srv import SetPhysicsProperties, GetPhysicsProperties
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Vector3
from xpp_msgs.msg import State6d,RobotStateJoint,StateLin3d, RobotStateCartesian
from geometry_msgs.msg import Vector3
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64
import rospy
from unitree_legged_msgs.msg import MotorCmd,MotorState
from sensor_msgs.msg import Imu
from geometry_msgs.msg import WrenchStamped

from logging import info
import time
import numpy as np

from pyunitree.robots.a1.constants import HIP_OFFSETS
from pyunitree.utils.kinematics import FootPositionInHipFrame, QuaternionToEulerMatrix

from pyunitree.base.daemon import Daemon
from multiprocessing import Manager

SIM_BASE_TIME_STEP = 0.001 # base timestep in Gazebo
SIM_BASE_RATE = 1000 # base timestep in Gazebo
    
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

    def __init__(self,updateRate = -1,name = "Gazebo state listener",publishXPP = False):
        super(GazeboInterface,self).__init__(updateRate,name)
        
        self.parser = GazeboMsgParser()
        self.jointNames = None
        self.position = [0]*12
        self.velocity = [0]*12
        self.footForces = [[0]*3]*4
        self.imu = [0]*10
        self.time = 0
        self.initTime = 0
        self.baseState = [0]*7
        self.baseIdx = -1
        self.footPos = np.zeros(12)
        self.footNames = ["a1_gazebo::FL_foot","a1_gazebo::FR_foot","a1_gazebo::RL_foot","a1_gazebo::RR_foot"]
        self.footIdx = []


        self.publishXPP = publishXPP
        if self.publishXPP:
            self.jointState,self.robotState = RobotStateJoint(), RobotStateCartesian()
            self.jointState.joint_state.name = self.motorNames
            self.robotState.ee_motion = [StateLin3d(),StateLin3d(),StateLin3d(),StateLin3d()]
            self.robotState.ee_forces = [Vector3(),Vector3(),Vector3(),Vector3()]
            self.jointState.ee_contact = [False]*4
            self.robotState.ee_contact = [False]*4

        self.__shared = Manager().Namespace()
        self.__shared.cmd = [0]*60

        self.sharedStateSize = 39
        self.sharedStateType = np.float32

        self.stateIsValid = RawValue("b",False)
        self.rawRemotePtr = None
        self.initSharedStateArray(self.sharedStateSize,"RobotState")

    def processInit(self):
        self.node = rospy.init_node('unitreepy_node')
    
        self.imuSub = rospy.Subscriber("/trunk_imu", Imu, self.imuVectorCallback)

        self.footForceSubs = [rospy.Subscriber(name, WrenchStamped, self.footForceVectorCallback,(idx)) 
                                                            for idx,name in enumerate(GazeboInterface.footContactNames)]

        self.motorSubs = [rospy.Subscriber(name+"/state", MotorState, self.motorVectorCallback,(idx)) 
                                                            for idx,name in enumerate(GazeboInterface.controllerNames)]

        self.servoPublishers = [rospy.Publisher(controllerName+"/command", MotorCmd,queue_size=0) 
                                                            for controllerName in GazeboInterface.controllerNames]
        if self.publishXPP:
            self.baseStateSub = rospy.Subscriber("/gazebo/link_states", LinkStates, self.baseStateCallback)
            self.jointPublisher = rospy.Publisher("/xpp/joint_a1_des", RobotStateJoint,queue_size=10) 
            self.robotPublisher = rospy.Publisher("/xpp/state_des", RobotStateCartesian,queue_size=10) 

        self.initTime = rospy.get_time()
        info("Unitreepy Gazebo listener: Attempting to receive initial position from Gazebo")
        while not self.stateIsValid.value:
            time.sleep(0.001)
        info("Unitreepy Gazebo listener: Initial state received")

    def publishXPPCallback(self):
        base = State6d()
        base.pose.position.x = self.baseState[0]
        base.pose.position.y = self.baseState[1]
        base.pose.position.z = self.baseState[2]

        base.pose.orientation.x  = self.baseState[3]
        base.pose.orientation.y  = self.baseState[4]
        base.pose.orientation.z  = self.baseState[5]
        base.pose.orientation.w  = self.baseState[6]
        
        self.robotState.base = base
        self.jointState.base = base
        self.jointState.joint_state.position = self.position

        for i in range(4):
            self.robotState.ee_motion[i].pos.x = self.footPos[3*i]
            self.robotState.ee_motion[i].pos.y = self.footPos[3*i+1]
            self.robotState.ee_motion[i].pos.z = self.footPos[3*i+2]

            self.robotState.ee_forces[i].x = self.footForces[i][0]
            self.robotState.ee_forces[i].y = self.footForces[i][1]
            self.robotState.ee_forces[i].z = self.footForces[i][2]
            
            self.jointState.ee_contact[i] = self.footForces[i][2]>9
            self.robotState.ee_contact[i] = self.footForces[i][2]>9
        

        self.jointPublisher.publish(self.jointState)
        self.robotPublisher.publish(self.robotState)


    def action(self):
        try:
            if self.publishXPP:
                self.publishXPPCallback()
            command = self.__shared.cmd
            self.time = rospy.get_time()-self.initTime
            self.rawStateBuffer[38] = self.time
            self.sendCommand(command)
            return True
        except BrokenPipeError:
            return False
        except ConnectionResetError:
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
    
    def baseStateCallback(self,msg):
        if self.baseIdx <0:
            self.baseIdx = msg.name.index("a1_gazebo::trunk")
        
        pose = msg.pose[self.baseIdx]
        self.baseState[0] = pose.position.x
        self.baseState[1] = pose.position.y
        self.baseState[2] = pose.position.z

        self.baseState[3] = pose.orientation.x
        self.baseState[4] = pose.orientation.y
        self.baseState[5] = pose.orientation.z
        self.baseState[6] = pose.orientation.w
        
        mat = QuaternionToEulerMatrix(self.baseState[3:])
        for i in range(4):
            self.footPos[3*i:3*i+3] = (FootPositionInHipFrame(self.position[3*i:3*i+3],l_hip_sign=i)+HIP_OFFSETS[i])@mat+self.baseState[:3]

    def motorVectorCallback(self,msg,idx):
        self.rawStateBuffer[14+idx] = msg.q
        self.rawStateBuffer[26+idx] = msg.dq
        
        #self.position[idx] = msg.q
        #self.velocity[idx] = msg.dq

    def imuVectorCallback(self,msg):
        self.parser.writeImuToBuffer(msg,self.rawStateBuffer)
        self.stateIsValid.value = 1

    def footForceVectorCallback(self,msg,footIdx):
        self.parser.writeFootForcesToBuffer(msg,self.rawStateBuffer,footIdx)

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
        initial_position = np.copy(self.rawStateBuffer[14:26])
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

    def slowDownSim(self,sim_slowdown=1):        
        set_physics = rospy.ServiceProxy('/gazebo/set_physics_properties', SetPhysicsProperties)
        get_physics = rospy.ServiceProxy('/gazebo/get_physics_properties', GetPhysicsProperties)
        # Slowdown by decreasing update rate
        time_step = Float64(SIM_BASE_TIME_STEP) #max_time_step
        max_update_rate = Float64(SIM_BASE_RATE/sim_slowdown)
        gravity = Vector3()
        gravity.x = 0.0
        gravity.y = 0.0
        gravity.z = -9.8
        ode_config = get_physics().ode_config
        set_physics(time_step.data, max_update_rate.data, gravity, ode_config)

    def resetPose(self):
        rospy.ServiceProxy("/gazebo/reset_world",Empty)()

    def stop(self):
        self.cleanup()