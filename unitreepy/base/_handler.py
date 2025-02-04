from numpy import array
import numpy as np
from time import perf_counter, sleep
from multiprocessing import Process, Manager, RawArray
from types import SimpleNamespace

from unitreepy.base.daemon import Daemon

from ..parsers.low_level import LowLevelParser
from ..utils._pos_profiles import p2p_cos_profile
from ..robots._default.constants import POSITION_GAINS, DAMPING_GAINS, INIT_ANGLES
from .daemon import Daemon
from unitreepy.interfaces.shared_state import SHM_IMPORTED

CONSTANTS = SimpleNamespace()

CONSTANTS.POSITION_GAINS = POSITION_GAINS
CONSTANTS.DAMPING_GAINS = DAMPING_GAINS
CONSTANTS.INIT_ANGLES = INIT_ANGLES


class RobotHandler(LowLevelParser, Daemon):
    """Creating the Robot Handler
       to bind the specific interface through """

    def __init__(self, update_rate=1000, constants=CONSTANTS):
        LowLevelParser.__init__(self)
        Daemon.__init__(self, update_rate, "RobotHandler")

        # create the service namespace, to store incoming comands and
        self.__shared = Manager().Namespace()
        self.__shared.command = self._zero_command
        self.__shared.process_is_working = False

        self.high_interface = None
        self.set_gains(position_gains=constants.POSITION_GAINS,
                       damping_gains=constants.DAMPING_GAINS)

        # Shared array init
        self.init_shared_state_array(39, "a1.robot_state")
        self.init_wireless_remote_array()

        self.__copy_state()

    def init_wireless_remote_array(self):
        name = "a1.wireless_remote"
        remote_byte_size=320
        data = np.zeros(remote_byte_size,dtype=np.bytes_)
        self.shared_state.register_shared_memory(name,remote_byte_size,np.bytes_)
        self.raw_remote_ptr = np.frombuffer(self.shared_state[name], dtype=np.float32)
        np.copyto(self.raw_remote_ptr, np.frombuffer(data, dtype=np.float32))

    def process_init(self):
        self.init_time = perf_counter()

    def on_start(self):
        print('Robot moving to initial position...')

        # /////////// REWRITE THIS ////////////////
        command = self._zero_command
        for i in range(5):
            self.__send_command(command)
            self.__update_state()
            sleep(0.001)

        terminal_time = 3
        initial_position = array(self.joint_angles)
        self.init_time = perf_counter()
        actual_time = 0
        desired_position = array(CONSTANTS.INIT_ANGLES)

        while actual_time <= terminal_time:
            if self.__shared.process_is_working:
                break

            actual_time = perf_counter() - self.init_time
            position, _ = p2p_cos_profile(actual_time,
                                          initial_pose=initial_position,
                                          final_pose=desired_position,
                                          terminal_time=terminal_time)

            command = self.set_angles(position)
            self.__send_command(command)
            self.__update_state()
        # //////////////////////////////////////////////////////////

    def action(self):
        command = self.__shared.command
        self.__send_command(command)
        low_state = self.__update_state()
        return True

    def __update_state(self):
        # update state based on incoming data
        low_state = self.__receive_state()
        self.parse_state(low_state)
        self.__copy_state()
        return low_state

    def __receive_state(self):
        # receive the low state and parse
        low_state = self.receiver()
        return low_state

    def __copy_state(self):
        compressed_state = np.hstack([self.quaternion, self.gyro, self.accelerometer,
                                    self.foot_force, self.joint_angles, self.joint_speed, [self.tick/1000]])

        if self.quaternion is None:
            return

        np.copyto(self.raw_remote_ptr,np.frombuffer(np.array(self.wireless_remote),dtype=np.float32))
        np.copyto(self.raw_state_buffer, compressed_state)


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

    def set_states(self,
                   desired_position,
                   desired_velocity,
                   desired_torque):

        command = self.build_command(desired_position=desired_position,
                                     desired_vel=desired_velocity,
                                     desired_torque=desired_torque,
                                     position_gains=self.position_gains,
                                     damping_gains=self.damping_gains)

        self.__shared.command = command

        return command

    def set_command(self, command):
        self.__shared.command = command

    def send(self, command):
        self.__shared.command = command

    def set_torques(self, desired_torque):
        """Build the command from desired torque"""
        command = self.build_command(desired_torque=desired_torque)
        self.__shared.command = command
        return command

    def set_angles(self, desired_position):
        '''Build the command from desired position'''
        command = self.build_command(desired_position=desired_position,
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
        initial_position = np.copy(self.raw_state_buffer[14:26])
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


    # def get
