from numpy import array
import numpy as np
from time import perf_counter, sleep
from multiprocessing import Process, Manager, RawArray
from types import SimpleNamespace

from ..parsers.low_level import LowLevelParser
from ..utils._pos_profiles import p2p_cos_profile
from ..robots._default.constants import POSITION_GAINS, DAMPING_GAINS, INIT_ANGLES


CONSTANTS = SimpleNamespace()

CONSTANTS.POSITION_GAINS = POSITION_GAINS
CONSTANTS.DAMPING_GAINS = DAMPING_GAINS
CONSTANTS.INIT_ANGLES = INIT_ANGLES


class RobotHandler(LowLevelParser):
    """Creating the Robot Handler
       to bind the specific interface through """

    def __init__(self, update_rate=1000, constants = CONSTANTS):
        LowLevelParser.__init__(self)
        self.update_rate = update_rate
        self.state = Manager().Namespace()
        self.state.time = 0
        # create the service namespace, to store incoming comands and
        self.__shared = Manager().Namespace()
        self.__shared.command = self._zero_command
        self.__shared.process_is_working = False 

        self.set_gains(position_gains=constants.POSITION_GAINS,
                       damping_gains=constants.DAMPING_GAINS)


        #Shared array init
        self.rawState = RawArray("f",39)
        data = np.zeros(39)
        rawState = np.frombuffer(self.rawState, dtype=np.float32)
        np.copyto(rawState, data)
        self.__copy_state()

        self._handler_process = Process(target=self.__handler)

    def __copy_state(self):
        # copy the internal states to shared memory
        self.state.joint_angles = self.joint_angles
        self.state.joint_speed = self.joint_speed
        self.state.joint_torques = self.joint_torques

        self.state.accel = self.accelerometer
        self.state.quaternion = self.quaternion
        self.state.gyro = self.gyro
        self.state.ticker = self.tick

        self.state.footforce = self.foot_force
        self.state.footforceEst = self.foot_force_est


        # footforce = np.array([force[2] for force in self.foot_force])

        footforce = self.foot_force

        compressedState = np.hstack([self.quaternion,self.gyro,self.accelerometer,footforce,self.joint_angles,self.joint_speed,[self.tick]])

        if self.quaternion == None:
            return
        rawState = np.frombuffer(self.rawState, dtype=np.float32)
        np.copyto(rawState, compressedState)


    def __update_state(self):
        # update state based on incoming data
        self.__receive_state()
        self.__copy_state()

    def __receive_state(self):
        # receive the low state and parse
        low_state = self.receiver()
        self.parse_state(low_state)

    def __send_command(self, command):
        # send low level command through transmitter
        self.transmitter(command)

    def set_transmitter(self, transmitter):
        self.transmitter = transmitter

    def set_receiver(self, receiver):
        self.receiver = receiver

    def bind_interface(self, receiver, transmitter):
        self.set_transmitter(transmitter)
        self.set_receiver(receiver)
    
    def __del__(self):
        self.stop(output=True)

    def start(self):
        print('Robot moving to initial position...')
        
        # /////////// REWRITE THIS ////////////////
        command = self._zero_command
        for i in range(5):
            self.__send_command(command)
            self.__update_state()
            sleep(0.001)

        terminal_time = 3
        initial_position = array(self.state.joint_angles)
        init_time = perf_counter()
        actual_time = 0
        desired_position = array(CONSTANTS.INIT_ANGLES)

        while actual_time <= terminal_time:
            if self.__shared.process_is_working:
                break
            
            actual_time = perf_counter() - init_time
            position, _ = p2p_cos_profile(actual_time,
                                          initial_pose=initial_position,
                                          final_pose=desired_position,
                                          terminal_time=terminal_time)

            command = self.set_angles(position)
            self.__send_command(command)
            self.__update_state()
        # //////////////////////////////////////////////////////////

        print('Robot in initial position....')
        self._handler_process.start()
        print('Waiting for process to start...')
        sleep(0.2)


    def stop(self, output=False):
        self._handler_process.terminate()
        if output:
            print('Robot process was terminated')
        
    def __handler(self):

        try:
            self.__shared.process_is_working = True
            # print(self.state.ticker)
            initial_time = perf_counter()
            tick = 0
            while True:
                actual_time = perf_counter() - initial_time
                command = self.__shared.command 
                self.state.time = actual_time
                if actual_time - tick >= 1/self.update_rate:
                    self.__send_command(command)
                    self.__update_state()
                    tick = actual_time

        except KeyboardInterrupt:
            print('Exit')


    def set_states(self,
                   desired_position,
                   desired_velocity,
                   desired_torque):

        command = self.build_command(desired_pos=desired_position,
                                     desired_vel=desired_velocity,
                                     desired_torque=desired_torque,
                                     position_gains=self.position_gains,
                                     damping_gains=self.damping_gains)

        self.__shared.command = command

        return command

    def set_torques(self, desired_torque):
        """Build the command from desired torque"""
        command = self.build_command(desired_torque=desired_torque)
        self.__shared.command = command
        return command

    def set_angles(self, desired_position):
        '''Build the command from desired position'''
        command = self.build_command(desired_pos=desired_position,
                                     position_gains=self.position_gains,
                                     damping_gains=self.damping_gains)
        self.__shared.command = command
        return command


    def set_gains(self, position_gains, damping_gains, output=False):
        self.position_gains = array(position_gains)
        self.damping_gains = array(damping_gains)
        if output:
            print('New gains were setted')

    def set_update_rate(self, update_rate=1000):
        self.update_rate = update_rate


    def move_to(self, desired_position, terminal_time=3):
        """Move to desired position with poitn to point """
        initial_position = array(self.state.joint_angles)
        init_time = perf_counter()
        actual_time = 0

        while actual_time <= terminal_time:
            actual_time = perf_counter() - init_time
            position, _ = p2p_cos_profile(actual_time,
                                          initial_pose=initial_position,
                                          final_pose=desired_position,
                                          terminal_time=terminal_time)
            self.set_angles(position)
                

    def move_to_init(self):
        self.move_to(array(CONSTANTS.INIT_ANGLES))


