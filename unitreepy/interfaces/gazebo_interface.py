from gazebo_msgs.srv import SetPhysicsProperties, GetPhysicsProperties
from gazebo_msgs.msg import LinkStates
from geometry_msgs.msg import Vector3
from geometry_msgs.msg import Vector3
from std_msgs.msg import Float64

import rospy
from sensor_msgs.msg import Imu
from geometry_msgs.msg import WrenchStamped
from unitree_legged_msgs.msg import MotorCmd,MotorState

from unitreepy.utils.kinematics import foot_position_hip_frame, quat_to_euler_matrix
from unitreepy.parsers.gazebo import GazeboMsgParser
from unitreepy.robots.a1.constants import HIP_OFFSETS
from unitreepy.robots.a1.constants import INIT_ANGLES,POSITION_GAINS,DAMPING_GAINS
from unitreepy.utils._pos_profiles import p2p_cos_profile
from unitreepy.base.daemon import Daemon

from multiprocessing import Manager
from multiprocessing.sharedctypes import RawValue

from logging import info
import time
import numpy as np

SIM_BASE_TIME_STEP = 0.001 # base timestep in Gazebo
SIM_BASE_RATE = 1000 # base timestep in Gazebo
    
class GazeboInterface(Daemon):
    controller_names  =  [
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

    foot_contact_names = [ 
                        "/visual/FR_foot_contact/the_force",
                        "/visual/FL_foot_contact/the_force",
                        "/visual/RR_foot_contact/the_force",
                        "/visual/RL_foot_contact/the_force"
                        ]

    #Joint names from JointState message
    motor_names       = [
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
    motor_mappings = {name:i for i,name in enumerate(motor_names)}

    def __init__(self,update_rate = -1,name = "Gazebo state listener",publish_xpp = False):
        super(GazeboInterface,self).__init__(update_rate,name)
        
        self.parser = GazeboMsgParser()
        self.joint_names = None
        self.position = [0]*12
        self.velocity = [0]*12
        self.foot_forces = [[0]*3]*4
        self.imu = [0]*10
        self.time = 0
        self.init_time = 0
        self.base_state = [0]*7
        self.base_idx = -1
        self.foot_pos = np.zeros(12)
        self.foot_names = ["a1_gazebo::FL_foot","a1_gazebo::FR_foot","a1_gazebo::RL_foot","a1_gazebo::RR_foot"]
        self.foot_idx = []


        self.publish_xpp = publish_xpp
        if self.publish_xpp:
            self.joint_state,self.robot_state = RobotStateJoint(), RobotStateCartesian()
            self.joint_state.joint_state.name = self.motor_names
            self.robot_state.ee_motion = [StateLin3d(),StateLin3d(),StateLin3d(),StateLin3d()]
            self.robot_state.ee_forces = [Vector3(),Vector3(),Vector3(),Vector3()]
            self.joint_state.ee_contact = [False]*4
            self.robot_state.ee_contact = [False]*4

        self.__shared = Manager().Namespace()
        self.__shared.cmd = [0]*60

        self.shared_state_type = np.float32

        self.state_is_valid = RawValue("b",False)
        self.raw_remote_ptr = None
        self.init_shared_state_array(39,"a1.robot_state")

    def process_init(self):
        self.node = rospy.init_node('unitreepy_node')
    
        self.imu_sub = rospy.Subscriber("/trunk_imu", Imu, self.imu_vector_callback)

        self.foot_force_subs = [rospy.Subscriber(name, WrenchStamped, self.foot_force_vector_callback,(idx)) 
                                                            for idx,name in enumerate(GazeboInterface.foot_contact_names)]

        self.motor_subs = [rospy.Subscriber(name+"/state", MotorState, self.motor_vector_callback,(idx)) 
                                                            for idx,name in enumerate(GazeboInterface.controller_names)]

        self.servo_publishers = [rospy.Publisher(controller_name+"/command", MotorCmd,queue_size=0) 
                                                            for controller_name in GazeboInterface.controller_names]
        if self.publish_xpp:
            from xpp_msgs.msg import State6d,RobotStateJoint,StateLin3d, RobotStateCartesian
            self.base_state_sub = rospy.Subscriber("/gazebo/link_states", LinkStates, self.base_state_callback)
            self.joint_publisher = rospy.Publisher("/xpp/joint_a1_des", RobotStateJoint,queue_size=10) 
            self.robot_publisher = rospy.Publisher("/xpp/state_des", RobotStateCartesian,queue_size=10) 

        self.init_time = rospy.get_time()
        info("Unitreepy Gazebo listener: Attempting to receive initial position from Gazebo")
        while not self.state_is_valid.value:
            time.sleep(0.001)
        info("Unitreepy Gazebo listener: Initial state received")

    def publish_xpp_callback(self):
        base = State6d()
        base.pose.position.x = self.base_state[0]
        base.pose.position.y = self.base_state[1]
        base.pose.position.z = self.base_state[2]

        base.pose.orientation.x  = self.base_state[3]
        base.pose.orientation.y  = self.base_state[4]
        base.pose.orientation.z  = self.base_state[5]
        base.pose.orientation.w  = self.base_state[6]
        
        self.robot_state.base = base
        self.joint_state.base = base
        self.joint_state.joint_state.position = self.position

        for i in range(4):
            self.robot_state.ee_motion[i].pos.x = self.foot_pos[3*i]
            self.robot_state.ee_motion[i].pos.y = self.foot_pos[3*i+1]
            self.robot_state.ee_motion[i].pos.z = self.foot_pos[3*i+2]

            self.robot_state.ee_forces[i].x = self.foot_forces[i][0]
            self.robot_state.ee_forces[i].y = self.foot_forces[i][1]
            self.robot_state.ee_forces[i].z = self.foot_forces[i][2]
            
            self.joint_state.ee_contact[i] = self.foot_forces[i][2]>9
            self.robot_state.ee_contact[i] = self.foot_forces[i][2]>9
        

        self.joint_publisher.publish(self.joint_state)
        self.robot_publisher.publish(self.robot_state)


    def action(self):
        try:
            if self.publish_xpp:
                self.publish_xpp_callback()
            command = self.__shared.cmd
            self.time = rospy.get_time()-self.init_time
            self.raw_state_buffer[38] = self.time
            self.send_command(command)
            return True
        except BrokenPipeError:
            return False
        except ConnectionResetError:
            return False
    
    def on_start(self):
        while not self.state_is_valid.value:
            time.sleep(0.01)
        
    #Utils
    
    def send_command(self,command):
        motor_cmd = MotorCmd()
        for motorId in range(12):
            motor_cmd.mode = 0x0A
            motor_cmd.q=command[motorId * 5]
            motor_cmd.Kp=command[motorId * 5+1]
            motor_cmd.dq=command[motorId * 5+2]
            motor_cmd.Kd=command[motorId * 5+3]
            motor_cmd.tau=command[motorId * 5+4]
            self.servo_publishers[motorId].publish(motor_cmd)

    def build_command(self,
                      desired_position=np.zeros(12),
                      desired_vel=np.zeros(12),
                      desired_torque=np.zeros(12),
                      position_gains=np.zeros(12),
                      damping_gains=np.zeros(12)):

        command = np.zeros(60)

        for motor_id in range(12):
            command[motor_id * 5] = desired_position[motor_id]
            command[motor_id * 5 + 1] = position_gains[motor_id]
            command[motor_id * 5 + 2] = desired_vel[motor_id]
            command[motor_id * 5 + 3] = damping_gains[motor_id]
            command[motor_id * 5 + 4] = desired_torque[motor_id]

        return command
    
    def base_state_callback(self,msg):
        if self.base_idx <0:
            self.base_idx = msg.name.index("a1_gazebo::trunk")
        
        pose = msg.pose[self.base_idx]
        self.base_state[0] = pose.position.x
        self.base_state[1] = pose.position.y
        self.base_state[2] = pose.position.z

        self.base_state[3] = pose.orientation.x
        self.base_state[4] = pose.orientation.y
        self.base_state[5] = pose.orientation.z
        self.base_state[6] = pose.orientation.w
        
        mat = quat_to_euler_matrix(self.base_state[3:])
        for i in range(4):
            self.foot_pos[3*i:3*i+3] = (foot_position_hip_frame(self.position[3*i:3*i+3],l_hip_sign=i)+HIP_OFFSETS[i])@mat+self.base_state[:3]

    def motor_vector_callback(self,msg,idx):
        self.raw_state_buffer[14+idx] = msg.q
        self.raw_state_buffer[26+idx] = msg.dq
        
        #self.position[idx] = msg.q
        #self.velocity[idx] = msg.dq

    def imu_vector_callback(self,msg):
        self.parser.write_imu_to_buffer(msg,self.raw_state_buffer)
        self.state_is_valid.value = 1

    def foot_force_vector_callback(self,msg,foot_idx):
        self.parser.write_foot_forces_to_buffer(msg,self.raw_state_buffer,foot_idx)

    def send(self,cmd):
        self.__shared.cmd = cmd
    
    def set_command(self,cmd):
        self.__shared.cmd = cmd
        
    def set_torques(self,desired_torques):
        self.send(self.build_command(desired_torque=desired_torques))

    def set_angles(self,desired_position):
        self.send(self.build_command(desired_position=desired_position,position_gains=POSITION_GAINS,damping_gains=DAMPING_GAINS))

    def move_to(self, desired_position, terminal_time=3):
        """Move to desired position with poitn to point """
        initial_position = np.copy(self.raw_state_buffer[14:26])
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

    def slow_down_sim(self,sim_slowdown=1):        
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

    def reset_pose(self):
        rospy.ServiceProxy("/gazebo/reset_world",Empty)()