from pyunitree.parsers.gazebo import GazeboMsgParser
from pyunitree.robots.a1.constants import INIT_ANGLES,POSITION_GAINS,DAMPING_GAINS
from pyunitree.utils._pos_profiles import p2p_cos_profile
import rospy
from unitree_legged_msgs.msg import LowState,MotorCmd,MotorState
from sensor_msgs.msg import Imu,JointState
from geometry_msgs.msg import WrenchStamped

from multiprocessing import Process, Manager
from logging import info
import time
import numpy as np

from scipy.spatial.transform import Rotation as R

class GazeboInterface:
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

    def __init__(self,update_rate):
        self.parser = GazeboMsgParser()
        self.lastState = LowState()
        self.update_rate = update_rate

        self.jointNames = None
        self.position = [0]*12
        self.velocity = [0]*12
        self.footForces = [[0]*3]*4
        self.imu = [0]*10
        self.time = 0
        self.initTime = 0

        self.__shared = Manager().Namespace()
        self.__shared.joint_angles = [0]*12
        self.__shared.joint_speed = [0]*12
        self.__shared.footforce =  [0]*4
        self.__shared.footForces = [[0]*3]*4
        self.__shared.imu = [0]*10
        self.__shared.jointNames = None
        self.__shared.ticker = 0

        self.__shared.quaternion = self.imu[:4]
        self.__shared.gyro = self.imu[4:7]
        self.__shared.accel = self.imu[7:10]

        self.__shared.handlerIsWorking = False
        self.__shared.cmd = [0]*60

    
    def sta(self):
        return self.__shared.position

    def start(self):
        self.handlerProc = Process(target=self.__handler,daemon=True)
        self.handlerProc.start()

        while not self.__shared.handlerIsWorking:
            time.sleep(0.01)

    def __del__(self):
        self.stop()
        
    def stop(self):
        self.handlerProc.terminate()
        
    def __handler(self):
        try:
            self.node = rospy.init_node('unitreepy_node')
        
            self.imuSub = rospy.Subscriber("/trunk_imu", Imu, self.imuVectorCallback)

            self.footForceSubs = [rospy.Subscriber(name, WrenchStamped, self.footForceVectorCallback,(idx)) 
                                                                for idx,name in enumerate(GazeboInterface.footContactNames)]

            self.servoSubs = rospy.Subscriber("/a1_gazebo/joint_states", JointState, self.jointStatesVectorCallback) 
            
            self.motorSubs = [rospy.Subscriber(name+"/state", MotorState, self.motorVectorCallback,(idx)) 
                                                                for idx,name in enumerate(GazeboInterface.controllerNames)]

            self.servoPublishers = [rospy.Publisher(controllerName+"/command", MotorCmd) 
                                                                for controllerName in GazeboInterface.controllerNames]

            self.initTime = rospy.get_time()

            info("Unitreepy Gazebo listener: Attempting to receive initial position from Gazebo")
            while self.jointNames==None:
                time.sleep(0.001)
            info("Unitreepy Gazebo listener: Initial state received")
            
            self.__shared.handlerIsWorking = True

            initial_time = time.perf_counter()
            last_tick = 0
            while True:
                actual_time = time.perf_counter() - initial_time
                try:
                    command = self.__shared.cmd

                    if actual_time - last_tick >= 1/self.update_rate:
                        self.moveStateToShared()
                        self.sendCommand(command)
                        last_tick = actual_time
                        
                except BrokenPipeError:
                    break

        except KeyboardInterrupt:
            print('Exit')
    

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

    def getSharedState(self):
        return self.__shared

    def moveStateToShared(self):
        self.__shared.quaternion = self.imu[:4]
        self.__shared.gyro = self.imu[4:7]
        self.__shared.accel = self.imu[7:10]
        self.__shared.footForces = self.footForces
        self.__shared.joint_angles = self.position
        self.__shared.joint_speed = self.velocity
        self.__shared.jointNames = self.jointNames
        self.__shared.ticker = self.time

        self.__shared.footforce = [force[2] for force in self.footForces]
        
    def motorVectorCallback(self,msg,idx):
        self.position[idx] = msg.q
        self.velocity[idx] = msg.dq

    def imuVectorCallback(self,msg):
        self.imu = self.parser.vectorizeImuMsg(msg)
        self.time = rospy.get_time()-self.initTime
        self.jointNames = "A"

    def footForceVectorCallback(self,msg,footIdx):
        self.footForces[footIdx] = self.parser.vectorizeEeForce(msg)

    def jointStatesVectorCallback(self,msg):
        #self.position = msg.position
        #self.velocity = msg.velocity
        pass

    def send(self,cmd):
        self.__shared.cmd = cmd

    def receive(self):
        try:
            lowState = LowState()
            lowState = self.buildLowState()
        except:
            raise RuntimeError("Unitreepy Gazebo listener: Failed to build current state of the robot")

        self.lastState = lowState
        return lowState

    def buildLowState(self):
        lowState = LowState()

        position = self.__shared.position
        velocity = self.__shared.velocity
        jointNames = self.__shared.jointNames

        for i in range(4):
            jointIdx1 = GazeboInterface.motorMappings[jointNames[3*i]]
            jointIdx2 = GazeboInterface.motorMappings[jointNames[3*i+1]]
            jointIdx3 = GazeboInterface.motorMappings[jointNames[3*i+2]]

            lowState.motorState[jointIdx1].q = position[3*i]
            lowState.motorState[jointIdx2].q = position[3*i+1]
            lowState.motorState[jointIdx3].q = position[3*i+2]

            lowState.motorState[jointIdx1].dq = velocity[3*i]
            lowState.motorState[jointIdx2].dq = velocity[3*i+1]
            lowState.motorState[jointIdx3].dq = velocity[3*i+2]

        imu = self.__shared.imu
        lowState.imu = self.parser.parseImuVector(imu)

        footForces = self.__shared.footForces

        for i in range(4):
            lowState.eeForce[i],lowState.footForce[i] = self.parser.parseEeForceVector(footForces[i])
        
        lowState.tick = self.__shared.time

        return lowState

    def getPitchRoll(self):
        orientation = self.__shared.imu[:4]
        r = R.from_quat(orientation)
        rpy_angles = r.as_euler("xyz")
        return rpy_angles[1],rpy_angles[2]

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

