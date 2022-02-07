from logging import info
import numpy as np

from pyunitree.utils.kinematics import euler_from_quat,quat_to_euler_matrix
from pyunitree.robots.a1.constants import POSITION_GAINS,DAMPING_GAINS
from pyunitree.base.daemon import SHM_IMPORTED
from pyunitree.parsers.remote import WirelessRemote


class A1SharedState:
    #original motion imitation
    MPC_BODY_MASS = 108 / 9.8
    MPC_BODY_INERTIA = np.array((0.24, 0, 0, 0, 0.80, 0, 0, 0, 1.00))
    SHM_NAMES = ['raw_state_shm','raw_observer_shm','raw_model_shm','raw_remote_shm','raw_command_shm']
    # This is actually default foot positions in robot CS
    _DEFAULT_HIP_POSITIONS = (
        (0.17, -0.135, 0),
        (0.17, 0.135, 0),
        (-0.195, -0.135, 0),
        (-0.195, 0.135, 0),
    )
    
    '''
    MPC_BODY_MASS = 13.52
    MPC_BODY_INERTIA = np.array((0.032, 0, 0, 0, 0.283, 0, 0, 0, 0.308))
    _DEFAULT_HIP_POSITIONS = (
        (0.18, -0.14, 0),
        (0.18, 0.14, 0),
        (-0.18, -0.14, 0),
        (-0.18, 0.14, 0),
    )

    _DEFAULT_HIP_POSITIONS = (
        (0.1805, -0.047, 0),
        (0.1805, 0.047, 0),
        (-0.1805, -0.047, 0),
        (-0.1805, 0.047, 0),
    )

    '''

    
    KPS = POSITION_GAINS
    KDS = DAMPING_GAINS

    MPC_BODY_HEIGHT = 0.24
    MPC_VELOCITY_MULTIPLIER = 0.5
    FOOT_FORCE_THRESHOLD = 10
    
    DATA_TYPE = np.float32
    """
    self.raw_state_buffer
     quaternion  |4|
           gyro  |3|
          accel  |3|
     footforces  |4|
              q  |12|
             dq  |12|
            tick |1|
    """
    def __init__(self,state_buffer_ptr=None,model_buffer_ptr=None,observer_buffer_ptr=None,wireless_remote_ptr=None,command_ptr=None):

        if SHM_IMPORTED:
            from multiprocessing.shared_memory import SharedMemory
            self.raw_state_shm = SharedMemory(name="RobotState")
            self.raw_state_buffer = np.frombuffer(self.raw_state_shm.buf, dtype=self.DATA_TYPE)

            try:
                self.raw_model_shm = SharedMemory(name="Model")
                self.raw_model_buffer = np.frombuffer(self.raw_model_shm.buf, dtype=self.DATA_TYPE)
            except FileNotFoundError:
                self.raw_model_buffer = None

            try:
                self.raw_observer_shm = SharedMemory(name="Observer")
                self.raw_observer_buffer = np.frombuffer(self.raw_observer_shm.buf, dtype=self.DATA_TYPE)
            except:
                self.raw_observer_buffer = None

            try:
                self.raw_remote_shm = SharedMemory(name="WirelessRemote")
                self.raw_remote_ptr =  np.frombuffer(self.raw_remote_shm.buf,dtype=np.bytes_)
            except:
                self.raw_remote_ptr = None

            try:
                self.raw_command_shm = SharedMemory(create=True,name="Command",size=12)
                self.command_shm_created = True
            except FileExistsError:
                self.command_shm_created = False
                self.raw_command_shm = SharedMemory(name="Command")
            
            self.raw_command_buffer =  np.frombuffer(self.raw_command_shm.buf,dtype=np.float32)
            np.copyto(self.raw_command_buffer,np.zeros(3,dtype=np.float32))
        else:
            if state_buffer_ptr is not None:
                self.raw_state_buffer = np.frombuffer(state_buffer_ptr, dtype=self.DATA_TYPE)

            if model_buffer_ptr is not None:
                self.raw_model_buffer = np.frombuffer(model_buffer_ptr, dtype=self.DATA_TYPE)
            
            if observer_buffer_ptr is not None:
                self.raw_observer_buffer = np.frombuffer(observer_buffer_ptr, dtype=self.DATA_TYPE)

            if wireless_remote_ptr is not None:
                self.raw_remote_ptr = np.frombuffer(wireless_remote_ptr, dtype=np.float32)

            if command_ptr is not None:
                self.raw_command_buffer = np.frombuffer(command_ptr, dtype=np.float32)

        if wireless_remote_ptr is not None:
            self.wireless_parser = WirelessRemote()

        self.state_copy = np.zeros(self.raw_state_buffer.shape) if self.raw_state_buffer is not None else None
        self.model_copy = np.zeros(self.raw_model_buffer.shape) if self.raw_model_buffer is not None else None
        self.observer_copy = np.zeros(self.raw_observer_buffer.shape) if self.raw_observer_buffer is not None else None
        self.remote_copy = np.zeros(self.raw_remote_ptr.shape) if self.raw_remote_ptr is not None else None
        self.command_copy = np.zeros(self.raw_command_buffer.shape) if self.raw_command_buffer is not None else None

    def unlink_shared_memory(self):
        if SHM_IMPORTED: 
            for shm_name in self.SHM_NAMES:
                if hasattr(self, shm_name):
                    try:
                        getattr(self,shm_name).unlink()
                    except FileNotFoundError:
                        info("UNLINK: Shared memory {shm_name} block not found")

    def destroy_shared_memory(self):
        if SHM_IMPORTED: 
            for shm_name in self.SHM_NAMES:
                if hasattr(self, shm_name):
                    try:
                        getattr(self,shm_name).close()
                    except FileNotFoundError:
                        info("DESTROY: Shared memory {shm_name} block not found")

    def cleanup(self,destroy=False):
        self.unlink_shared_memory()
        if destroy:
            self.destroy_shared_memory()

    def copy_buffers(self,buffers=["state","model","observer","remote","command"]):
        if self.raw_state_buffer is not None and "state" in buffers:
            np.copyto(self.state_copy,self.raw_state_buffer)
        if self.raw_model_buffer is not None and "model" in buffers:
            np.copyto(self.model_copy,self.raw_model_buffer)
        if self.raw_observer_buffer is not None and "observer" in buffers:
            np.copyto(self.observer_copy,self.raw_observer_buffer)
        if self.raw_remote_ptr is not None and "remote" in buffers:
            np.copyto(self.remote_copy,self.raw_remote_ptr)
        if self.raw_command_buffer is not None and "command" in buffers:
            np.copyto(self.command_copy,self.raw_command_buffer)

    def get_state_buffer(self,use_cached=False):
        if use_cached:
            if self.state_copy is not None:
                return self.state_copy
            else:
                raise Exception("State buffer was not initialized in the instance of this shared state object")
        else:
            return self.raw_state_buffer

    def get_model_buffer(self,use_cached=False):
        if use_cached:
            if self.model_copy is not None:
                return self.model_copy
            else:
                raise Exception("Model buffer was not initialized in the instance of this shared state object")
        else:
            return self.raw_model_buffer

    def get_observer_buffer(self,use_cached=False):
        if use_cached:
            if self.observer_copy is not None:
                return self.observer_copy
            else:
                raise Exception("Observer buffer was not initialized in the instance of this shared state object")
        else:
            return self.raw_observer_buffer

    def get_remote_buffer(self,use_cached=False):
        if use_cached:
            if self.remote_copy is not None:
                return self.remote_copy
            else:
                raise Exception("Remote buffer was not initialized in the instance of this shared state object")
        else:
            return self.raw_remote_ptr
            
    def get_command_buffer(self,use_cached=False):
        if use_cached:
            if self.command_copy is not None:
                return self.command_copy
            else:
                raise Exception("Command buffer was not initialized in the instance of this shared state object")
        else:
            return self.raw_command_buffer

    def get_base_orientation_quaternion(self,use_cached=False):
        q = self.get_state_buffer(use_cached)[:4].copy()
        return [q[1],q[2],q[3],q[0]]
    
    def get_base_orientation_matrix(self,use_cached=False):
        return quat_to_euler_matrix(self.get_base_orientation_quaternion(use_cached=use_cached))

    def get_base_rpy(self,use_cached=False):
        return euler_from_quat(self.get_base_orientation_quaternion(use_cached=use_cached))

    def get_base_rpy_rate(self,use_cached=False):
        return self.get_state_buffer(use_cached)[4:7].copy()
    
    #NOTE: DO NOT USE ON REAL ROBOT GAZEBO ONLY
    #World space position and orientation of the robot's trunk
    def get_base_position_orientation_gazebo(self,use_cached=False): 
        return self.get_state_buffer(use_cached)[39:46].copy()
    #World space position of the robot's feet
    def get_foot_position_gazebo(self,use_cached=False): 
        return self.get_state_buffer(use_cached)[46:58].copy()

    def get_base_acceleration(self,use_cached=False):
        return self.get_state_buffer(use_cached)[7:10].copy()

    def get_foot_forces(self,use_cached=False):
        return self.get_state_buffer(use_cached)[10:14].copy()

    def get_joint_angles(self,use_cached=False):
        return self.get_state_buffer(use_cached)[14:26].copy()

    def get_motor_velocities(self,use_cached=False):
        return self.get_state_buffer(use_cached)[26:38].copy()
    
    def get_current_tick(self,use_cached=False):
        return self.get_state_buffer(use_cached)[38].copy()

    def get_foot_positions_base_frame(self,use_cached=False):
        #NOTE copy is important
        return self.get_model_buffer(use_cached)[36:].copy().reshape((4,3))
        
    def get_hip_positions_base_frame(self):
        return self._DEFAULT_HIP_POSITIONS

    def get_motor_position_gains(self):
        return self.KPS

    def get_motor_velocity_gains(self):
        return self.KDS

    def get_foot_contacts(self,use_cached=False):
        foot_force = self.get_foot_forces(use_cached)
        return foot_force > self.FOOT_FORCE_THRESHOLD
    
    def get_jacobians(self,use_cached=False):
        return self.get_model_buffer(use_cached)[:36].copy().reshape((12,3))
   
    def __get_base_state_estimate(self,use_cached=False):
        return self.get_observer_buffer(use_cached).copy().reshape((3,3))

    def get_base_velocity_world_frame(self):
        return self.__get_base_state_estimate()[0]
    
    def get_base_velocity_body_frame(self):
        return self.__get_base_state_estimate()[1]
        
    def get_base_position_world(self):
        return self.__get_base_state_estimate()[2]
    
    def set_command(self,command):
        np.copyto(self.raw_command_buffer,command)

    def get_command(self,use_cached=False):
        return self.get_command_buffer(use_cached).copy()

    def print_wireless_state(self):
        remote = self.wireless_parser
        state = self.raw_remote_ptr.copy().tobytes()
        remote.set_state(state)
        remote.update_state()
        print(f'\n{15*"/"} WIRELESS REMOTE STATE {15*"/"}')
        print(f' BUTTONS:\n A {remote.button.A} B {remote.button.B} X {remote.button.X} Y {remote.button.Y}',
            f'\n  Right {remote.button.right} Left {remote.button.left} Up {remote.button.up} Down {remote.button.down}',
            f'\n  R1 {remote.button.R1} R2 {remote.button.R2} L1 {remote.button.L1} L2 {remote.button.L2}',
            f'\n  Start {remote.button.start} Select {remote.button.select} F1 {remote.button.F1} F2 {remote.button.F2}',
            f'\n JOYSTICKS:\n LEFT: X {round(remote.left_joystick[0],2)} Y {round(remote.left_joystick[1],2)}',
            f'\n  RIGHT: X {round(remote.right_joystick[0],2)}  Y {round(remote.right_joystick[1],2)}',
            end=10*" " + "\n", flush=True)
        print(f'{52*"/"}\n')
        print(12*'\033[A', end="\r", flush=True)

    def print_full_state(self):
        print("get_base_orientation_quaternion")
        print(self.get_base_orientation_quaternion())
        print("get_base_acceleration")
        print(self.get_base_acceleration())
        print("get_foot_forces")
        print(self.get_foot_forces())
        print("get_joint_angles")
        print(self.get_joint_angles())
        print("get_motor_velocities")
        print(self.get_motor_velocities())
        print("get_base_velocity_world_frame")
        print(self.get_base_velocity_world_frame())
